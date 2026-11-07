import * as fs from "node:fs";
import * as path from "node:path";
import { afterAll, beforeAll, beforeEach, expect, test } from "vitest";
import * as zarr from "zarrita";
import { createSourceData } from "../src/io";
import { JazavReferenceStore } from "../src/jazav-store";
import { type ServedRequest, startFixtureServer } from "./fixture-server";

/**
 * Reads virtual OME-Zarr images, whose level 0 is a `jazav.reference` array, through
 * the store that resolves the references in the browser. Two fixtures: `jazav-v05` is
 * written by `scripts/make-local-fixtures.mjs` to make each rule of the format visible
 * in the pixels, and `jazav-tiff-grid` is the sample jazav's own writer produced.
 */
const FIXTURES = path.resolve(__dirname, "..", "..", "fixtures", "local");

let origin: string;
let requests: ServedRequest[];
let stop: () => Promise<void>;

beforeAll(async () => {
  const server = await startFixtureServer();
  origin = server.url;
  requests = server.requests;
  stop = server.close;
});

beforeEach(() => {
  requests.length = 0;
});

afterAll(async () => {
  await stop();
});

/** The gradient the generator writes, as `pixels()` in the fixture script defines it. */
function gradient(c: number, size: number): Uint8Array {
  const plane = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      plane[y * size + x] = (x * 4 + y * 2 + c * 61) % 256;
    }
  }
  return plane;
}

test("A virtual image opens like any other, with its level 0 read out of the raw files", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-v05/image.zarr` });

  expect(source.axis_labels).toEqual(["t", "c", "z", "y", "x"]);
  expect(source.loader).toHaveLength(2);
  // The shape of the array, not of the shard grid or of the stored blocks.
  expect(source.loader[0].shape).toEqual([1, 2, 1, 32, 32]);
  expect(source.names).toEqual(["Red channel", "Green channel"]);

  const { data, width, height } = await source.loader[0].getRaster({ selection: [0, 0, 0] });
  expect([width, height]).toEqual([32, 32]);
  // The blocks are stored out of order in the raw file; the table says where each goes.
  expect(data).toEqual(gradient(0, 32));
});

test("An inner chunk without a block reads as the array's fill value", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-v05/image.zarr` });
  const { data } = await source.loader[0].getRaster({ selection: [0, 1, 0] });

  const expected = gradient(1, 32);
  // Channel 1 has no block for its bottom-right 16x16 tile, and fill_value is 7.
  for (let y = 16; y < 32; y++) {
    expected.fill(7, y * 32 + 16, y * 32 + 32);
  }
  expect(data).toEqual(expected);
});

test("Reading a plane costs one table read per shard and one range request per block", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-v05/image.zarr` });
  requests.length = 0;

  await source.loader[0].getRaster({ selection: [0, 0, 0] });

  const tables = requests.filter((r) => r.path === "/jazav-v05/image.zarr/0/c/0/0/0/0/0");
  const blocks = requests.filter((r) => r.path === "/jazav-v05/raw/c0.bin");
  expect(tables).toHaveLength(1);
  expect(blocks).toHaveLength(4);
  // Each block is one 256-byte range, answered as a partial response rather than the file.
  expect(blocks.every((r) => r.status === 206)).toBe(true);
  expect(blocks.map((r) => r.range).sort()).toEqual([
    "bytes=320-575",
    "bytes=576-831",
    "bytes=64-319",
    "bytes=832-1087",
  ]);
  // Nothing of channel 1 was touched.
  expect(requests.some((r) => r.path.includes("c1.bin") || r.path.endsWith("/0/c/0/1/0/0/0"))).toBe(false);
});

test("The pyramid level above is a plain array and is read as one", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-v05/image.zarr` });
  requests.length = 0;

  const { data, width } = await source.loader[1].getRaster({ selection: [0, 0, 0] });

  expect(width).toBe(16);
  expect(data).toEqual(gradient(0, 16));
  expect(requests.map((r) => r.path)).toEqual(["/jazav-v05/image.zarr/1/c/0/0/0/0/0"]);
  expect(requests[0].range).toBeUndefined();
});

test("The jazav sample reads back exactly the bytes of its TIFF tiles", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-tiff-grid/tiff/virtual.zarr` });
  expect(source.loader).toHaveLength(3);
  expect(source.loader[0].shape).toEqual([1, 1, 128, 128]);

  const { data, width, height } = await source.loader[0].getRaster({ selection: [0, 0] });
  expect([width, height]).toEqual([128, 128]);

  // Each 64x64 uint16 tile is one uncompressed strip starting at byte 256 of its file;
  // what the store returns must be those bytes, placed on the grid the file names say.
  const expected = new Uint16Array(128 * 128);
  for (const [ty, tx] of [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
  ]) {
    const file = fs.readFileSync(path.join(FIXTURES, "jazav-tiff-grid", "raw", "tiff", `tile_y${ty}_x${tx}.tif`));
    const tile = new Uint16Array(file.buffer.slice(file.byteOffset + 256, file.byteOffset + 256 + 64 * 64 * 2));
    for (let y = 0; y < 64; y++) {
      expected.set(tile.subarray(y * 64, (y + 1) * 64), (ty * 64 + y) * 128 + tx * 64);
    }
  }
  expect(data).toEqual(expected);

  const blocks = requests.filter((r) => r.path.startsWith("/jazav-tiff-grid/raw/tiff/"));
  expect(blocks).toHaveLength(4);
  expect(new Set(blocks.map((r) => r.range))).toEqual(new Set(["bytes=256-8447"]));
});

test("The sample's compressed pyramid levels still read through the same store", async () => {
  const [source] = await createSourceData({ source: `${origin}/jazav-tiff-grid/tiff/virtual.zarr` });
  const { data, width, height } = await source.loader[2].getRaster({ selection: [0, 0] });

  expect([width, height]).toEqual([32, 32]);
  expect(Math.max(...data)).toBeGreaterThan(0);
});

test("A jazav:// prefix is accepted and opens the same image", async () => {
  const [source] = await createSourceData({ source: `jazav://${origin}/jazav-v05/image.zarr` });
  expect(source.loader[0].shape).toEqual([1, 2, 1, 32, 32]);
});

test("Metadata of an ordinary array passes through untouched", async () => {
  const plain = new zarr.FetchStore(`${origin}/`);
  const store = new JazavReferenceStore(plain);

  const [direct, wrapped] = await Promise.all([
    plain.get("/image-v05.zarr/0/zarr.json"),
    store.get("/image-v05.zarr/0/zarr.json"),
  ]);
  expect(wrapped).toEqual(direct);
  expect(await store.get("/image-v05.zarr/0/c/0/0/0/0/0")).toEqual(await plain.get("/image-v05.zarr/0/c/0/0/0/0/0"));
});

/** The real fixture, with its level-0 metadata replaced by `edit(metadata)`. */
function withEditedMetadata(image: string, edit: (metadata: Record<string, unknown>) => void) {
  const inner = new zarr.FetchStore(`${origin}/${image}`);
  const metadata = JSON.parse(fs.readFileSync(path.join(FIXTURES, image, "0", "zarr.json"), "utf8"));
  edit(metadata);
  const edited = new TextEncoder().encode(JSON.stringify(metadata));
  return new JazavReferenceStore({
    url: inner.url,
    get: (key, opts) => (key === "/0/zarr.json" ? Promise.resolve(edited) : inner.get(key, opts)),
  });
}

test("A format version this store does not read is refused by name", async () => {
  const store = withEditedMetadata("jazav-v05/image.zarr", (metadata) => {
    (metadata.codecs as Array<{ configuration: { version: number } }>)[0].configuration.version = 3;
  });
  await expect(zarr.open(zarr.root(store).resolve("0"), { kind: "array" })).rejects.toThrow(
    /jazav\.reference version 3, and this viewer reads version 2/,
  );
});

test("A source file whose size differs from the one the references were written for is refused", async () => {
  const store = withEditedMetadata("jazav-v05/image.zarr", (metadata) => {
    (metadata.codecs as Array<{ configuration: { sizes: number[] } }>)[0].configuration.sizes[0] = 1;
  });
  const array = await zarr.open(zarr.root(store).resolve("0"), { kind: "array" });
  await expect(zarr.get(array, [0, 0, 0, zarr.slice(0, 16), zarr.slice(0, 16)])).rejects.toThrow(
    /c0\.bin is 1088 bytes but the references were written for 1/,
  );
});
