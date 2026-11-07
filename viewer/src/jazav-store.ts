import { log } from "./logger";

/**
 * A store that resolves `jazav.reference` arrays in the browser.
 *
 * A virtual OME-Zarr written by [jazav](https://github.com/lorenzocerrone/jazav) is an
 * ordinary OME-Zarr whose level 0 holds no pixels. Each of its zarr chunks is a *shard*
 * holding a table of 20-byte references (source file, byte offset, byte length), one per
 * inner chunk, pointing into the original microscopy files, which sit next to the zarr
 * and are never modified. zarr-python reads it through a codec; zarrita has no hook a
 * codec could use to fetch ranges, so this does the same work one layer down, in the
 * store, where the key and the array's URL are both known:
 *
 * - every key is forwarded to the wrapped store, so pyramid levels, labels and tables
 *   read as before;
 * - a `zarr.json` whose only codec is `jazav.reference` is rewritten before zarrita
 *   sees it: the inner chunk shape becomes the chunk grid and the nested codec chain
 *   becomes the codecs, so zarrita believes it is reading a plain array of inner chunks;
 * - a chunk key of such an array is answered by reading the shard's table (once), taking
 *   the entry for the inner chunk, and issuing one HTTP range request against the
 *   source file.
 *
 * One request per table, then one range per inner chunk. The reference format is frozen
 * at version 2; see the jazav README for the byte layout this follows.
 */

/** The reference codec as it appears in level-0 metadata. */
const CODEC_NAME = "jazav.reference";
/** The format version this store reads. Anything else is refused rather than guessed at. */
const FORMAT_VERSION = 2;
/** `<IQQ`: uint32 source, uint64 offset, uint64 length. */
const ENTRY_BYTES = 20;
/** Twenty `0xFF` bytes mark an inner chunk without a block, which reads as `fill_value`. */
const FILL_SOURCE = 0xffffffff;

type AbsolutePath = `/${string}`;
type RangeQuery = { offset: number; length: number } | { suffixLength: number };

/** What this store needs from the store it wraps: zarrita's `AsyncReadable`, plus a URL. */
export interface ReadableWithUrl {
  url: string | URL;
  get(key: AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined>;
  getRange?(key: AbsolutePath, range: RangeQuery, opts?: RequestInit): Promise<Uint8Array | undefined>;
}

interface CodecMetadata {
  name: string;
  configuration?: Record<string, unknown>;
}

interface ReferenceCodecConfiguration {
  version: number;
  chunk_shape: number[];
  sources: string[];
  sizes?: number[];
  codecs: CodecMetadata[];
}

/** What is remembered about one reference array, keyed by its prefix in the store. */
interface ReferenceArray {
  /** The array's URL, which `sources` are relative to. */
  url: URL;
  /** The zarr chunk grid of the array as stored: one table per shard. */
  shardShape: number[];
  /** The shape of one stored block, which is what zarrita is told the chunk shape is. */
  chunkShape: number[];
  /** Inner chunks per shard along each axis. */
  chunksPerShard: number[];
  sources: string[];
  sizes: number[] | undefined;
  /** Decoded tables by shard key, or `undefined` for a shard key that does not exist. */
  tables: Map<string, Promise<DataView | undefined>>;
}

interface Reference {
  source: number;
  offset: number;
  length: number;
}

/** A chunk key under the default encoding with separator `/`: `<prefix>/c/<i>/<j>/...`. */
const CHUNK_KEY = /^(.*?)\/c((?:\/\d+)+)$/;

function isReferenceCodec(metadata: unknown): metadata is { codecs: [CodecMetadata] } {
  if (typeof metadata !== "object" || metadata === null) return false;
  const { node_type, codecs } = metadata as { node_type?: unknown; codecs?: unknown };
  return node_type === "array" && Array.isArray(codecs) && codecs.length === 1 && codecs[0]?.name === CODEC_NAME;
}

function parseConfiguration(codec: CodecMetadata, where: string): ReferenceCodecConfiguration {
  const configuration = codec.configuration as Partial<ReferenceCodecConfiguration> | undefined;
  if (configuration?.version !== FORMAT_VERSION) {
    throw new Error(
      `${where} uses ${CODEC_NAME} version ${String(configuration?.version)}, and this viewer reads version ${FORMAT_VERSION}.`,
    );
  }
  const { chunk_shape, sources, sizes, codecs } = configuration;
  if (!Array.isArray(chunk_shape) || !Array.isArray(sources) || !Array.isArray(codecs)) {
    throw new Error(`${where}: the ${CODEC_NAME} configuration needs chunk_shape, sources and codecs.`);
  }
  if (sizes !== undefined && (!Array.isArray(sizes) || sizes.length !== sources.length)) {
    throw new Error(`${where}: the ${CODEC_NAME} configuration lists ${sources.length} sources but sizes differs.`);
  }
  return { version: FORMAT_VERSION, chunk_shape, sources, sizes, codecs };
}

/** The URL a store key resolves to, the way `FetchStore` does it. */
function resolveKey(root: string | URL, key: AbsolutePath): URL {
  const base = new URL(root);
  if (!base.pathname.endsWith("/")) base.pathname += "/";
  const resolved = new URL(key.slice(1), base);
  resolved.search = base.search;
  return resolved;
}

function product(shape: number[]): number {
  return shape.reduce((a, b) => a * b, 1);
}

function withRange(init: RequestInit | undefined, offset: number, length: number): RequestInit {
  const headers = new Headers(init?.headers);
  headers.set("Range", `bytes=${offset}-${offset + length - 1}`);
  return { ...init, headers };
}

export class JazavReferenceStore implements ReadableWithUrl {
  /**
   * The wrapped store's URL, for callers that build URLs from the store they were
   * given. A plain property rather than a getter: the LRU wrapper reads it through a
   * Proxy, and a getter would then run with the Proxy as `this`, where a private field
   * cannot be read.
   */
  readonly url: string | URL;
  readonly #inner: ReadableWithUrl;
  /**
   * Every array whose metadata has passed through, reference or not, so that a chunk
   * key is classified once per array rather than once per chunk.
   */
  readonly #arrays = new Map<string, Promise<ReferenceArray | null>>();

  constructor(inner: ReadableWithUrl) {
    this.#inner = inner;
    this.url = inner.url;
  }

  async get(key: AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined> {
    if (key.endsWith("/zarr.json")) {
      return this.#getMetadata(key, opts);
    }
    const match = CHUNK_KEY.exec(key);
    if (match) {
      const array = await this.#array(match[1], opts);
      if (array) {
        const coords = match[2].slice(1).split("/").map(Number);
        return this.#getChunk(array, key, coords, opts);
      }
    }
    return this.#inner.get(key, opts);
  }

  async getRange(key: AbsolutePath, range: RangeQuery, opts?: RequestInit): Promise<Uint8Array | undefined> {
    const match = CHUNK_KEY.exec(key);
    if (match && (await this.#array(match[1], opts))) {
      // Nothing asks for part of a reference chunk today; answer it from the whole
      // chunk rather than refuse, so the store stays a complete `AsyncReadable`.
      const bytes = await this.get(key, opts);
      if (!bytes) return undefined;
      if ("suffixLength" in range) return bytes.slice(Math.max(0, bytes.length - range.suffixLength));
      return bytes.slice(range.offset, range.offset + range.length);
    }
    if (!this.#inner.getRange) {
      throw new Error("The wrapped store does not support range requests.");
    }
    return this.#inner.getRange(key, range, opts);
  }

  /**
   * Reads `<prefix>/zarr.json`, remembers whether it is a reference array, and hands
   * zarrita metadata it can act on: a reference array is presented as a plain array of
   * inner chunks, with the nested codec chain in place of the reference codec.
   */
  async #getMetadata(key: AbsolutePath, opts?: RequestInit): Promise<Uint8Array | undefined> {
    const prefix = key.slice(0, -"/zarr.json".length);
    const bytes = await this.#inner.get(key, opts);
    if (!bytes) {
      this.#arrays.set(prefix, Promise.resolve(null));
      return undefined;
    }
    let metadata: unknown;
    try {
      metadata = JSON.parse(new TextDecoder().decode(bytes));
    } catch {
      // Not ours to judge; zarrita reports malformed metadata better than we would.
      return bytes;
    }
    if (!isReferenceCodec(metadata)) {
      this.#arrays.set(prefix, Promise.resolve(null));
      return bytes;
    }

    const where = resolveKey(this.#inner.url, key).href;
    const configuration = parseConfiguration(metadata.codecs[0], where);
    const original = metadata as { chunk_grid?: { configuration?: { chunk_shape?: number[] } } };
    const shardShape = original.chunk_grid?.configuration?.chunk_shape;
    if (!shardShape || shardShape.length !== configuration.chunk_shape.length) {
      throw new Error(`${where}: chunk_grid.chunk_shape and the ${CODEC_NAME} chunk_shape must have the same rank.`);
    }
    const chunksPerShard = shardShape.map((shard, i) => {
      const inner = configuration.chunk_shape[i];
      if (shard % inner !== 0) {
        throw new Error(
          `${where}: shard shape ${shardShape} is not a multiple of chunk shape ${configuration.chunk_shape}.`,
        );
      }
      return shard / inner;
    });

    const array: ReferenceArray = {
      url: resolveKey(this.#inner.url, (prefix || "/") as AbsolutePath),
      shardShape,
      chunkShape: configuration.chunk_shape,
      chunksPerShard,
      sources: configuration.sources,
      sizes: configuration.sizes,
      tables: new Map(),
    };
    this.#arrays.set(prefix, Promise.resolve(array));
    log.debug("Reference array found", {
      url: array.url.href,
      shardShape,
      chunkShape: array.chunkShape,
      sources: array.sources.length,
    });

    const rewritten = {
      ...metadata,
      chunk_grid: { name: "regular", configuration: { chunk_shape: configuration.chunk_shape } },
      codecs: configuration.codecs,
    };
    return new TextEncoder().encode(JSON.stringify(rewritten));
  }

  /**
   * What is known about the array at `prefix`. zarrita needs an array's metadata before
   * it can name a chunk, so this is normally already answered; a chunk key arriving
   * first fetches the metadata itself rather than guessing.
   */
  #array(prefix: string, opts?: RequestInit): Promise<ReferenceArray | null> {
    let known = this.#arrays.get(prefix);
    if (!known) {
      known = this.#getMetadata(`${prefix}/zarr.json` as AbsolutePath, opts).then(
        () => this.#arrays.get(prefix) ?? null,
        (error) => {
          // Forget a failed lookup so a retry is possible, then let the chunk read fail.
          this.#arrays.delete(prefix);
          throw error;
        },
      );
      this.#arrays.set(prefix, known);
    }
    return known;
  }

  async #getChunk(
    array: ReferenceArray,
    key: AbsolutePath,
    coords: number[],
    opts?: RequestInit,
  ): Promise<Uint8Array | undefined> {
    if (coords.length !== array.chunkShape.length) {
      throw new Error(
        `Chunk key ${key} has ${coords.length} coordinates; the array has rank ${array.chunkShape.length}.`,
      );
    }
    const shardCoords = coords.map((c, i) => Math.floor(c / array.chunksPerShard[i]));
    // C-order index over the shard's inner chunk grid, which is the table's order.
    let index = 0;
    for (let i = 0; i < coords.length; i++) {
      index = index * array.chunksPerShard[i] + (coords[i] - shardCoords[i] * array.chunksPerShard[i]);
    }
    const table = await this.#table(array, key, shardCoords, opts);
    if (!table) {
      // A shard key that does not exist: every inner chunk of it reads as fill_value.
      return undefined;
    }
    const reference = readEntry(table, index);
    if (!reference) return undefined;
    return this.#fetchBlock(array, reference, opts);
  }

  /** The shard's table, fetched once however many of its inner chunks are read. */
  #table(
    array: ReferenceArray,
    chunkKey: AbsolutePath,
    shardCoords: number[],
    opts?: RequestInit,
  ): Promise<DataView | undefined> {
    const shardKey = `${chunkKey.slice(0, chunkKey.indexOf("/c/"))}/c/${shardCoords.join("/")}` as AbsolutePath;
    let table = array.tables.get(shardKey);
    if (!table) {
      const expected = ENTRY_BYTES * product(array.chunksPerShard);
      table = this.#inner.get(shardKey, opts).then((bytes) => {
        if (!bytes) return undefined;
        if (bytes.byteLength !== expected) {
          throw new Error(
            `The reference table ${resolveKey(this.#inner.url, shardKey).href} is ${bytes.byteLength} bytes; a shard of ${product(array.chunksPerShard)} chunks needs ${expected}.`,
          );
        }
        return new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      });
      table.catch(() => array.tables.delete(shardKey));
      array.tables.set(shardKey, table);
    }
    return table;
  }

  /** One range request for one stored block. */
  async #fetchBlock(array: ReferenceArray, reference: Reference, opts?: RequestInit): Promise<Uint8Array> {
    const path = array.sources[reference.source];
    if (path === undefined) {
      throw new Error(
        `A reference in ${array.url.href} names source ${reference.source}, which the array does not list.`,
      );
    }
    // Relative to the array's key prefix, as on disk; the raw files must be served in
    // the same layout relative to the zarr.
    const url = new URL(path, `${array.url.href}/`);
    const expectedSize = array.sizes?.[reference.source];
    const response = await fetch(url, withRange(opts, reference.offset, reference.length));

    if (response.status === 206) {
      const total = /\/(\d+)$/.exec(response.headers.get("Content-Range") ?? "")?.[1];
      if (total !== undefined && expectedSize !== undefined && Number(total) !== expectedSize) {
        throw new Error(
          `${url.href} is ${total} bytes but the references were written for ${expectedSize}; the source has been replaced or truncated.`,
        );
      }
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (bytes.byteLength !== reference.length) {
        throw new Error(`${url.href} returned ${bytes.byteLength} bytes for a ${reference.length}-byte range.`);
      }
      return bytes;
    }
    if (response.status === 200) {
      // The server ignored the Range header and sent the whole file. Costly, but usable
      // if it is the file the references describe.
      const bytes = new Uint8Array(await response.arrayBuffer());
      if (expectedSize !== undefined && bytes.byteLength !== expectedSize) {
        throw new Error(
          `${url.href} is ${bytes.byteLength} bytes but the references were written for ${expectedSize}; the source has been replaced or truncated.`,
        );
      }
      if (bytes.byteLength < reference.offset + reference.length) {
        throw new Error(
          `${url.href} is ${bytes.byteLength} bytes, too short for a block ending at ${reference.offset + reference.length}.`,
        );
      }
      return bytes.slice(reference.offset, reference.offset + reference.length);
    }
    if (response.status === 404) {
      throw new Error(`${url.href} was not found; a virtual image needs its source files served next to it.`);
    }
    throw new Error(`Unexpected response ${response.status} ${response.statusText} for ${url.href}`);
  }
}

/** The entry at `index`, or `undefined` for the fill marker. */
function readEntry(table: DataView, index: number): Reference | undefined {
  const at = index * ENTRY_BYTES;
  const source = table.getUint32(at, true);
  if (source === FILL_SOURCE) return undefined;
  // Offsets past 2^53 cannot be represented, nor requested from a browser; say so
  // rather than round silently.
  const offset = table.getBigUint64(at + 4, true);
  const length = table.getBigUint64(at + 12, true);
  if (offset > BigInt(Number.MAX_SAFE_INTEGER) || length > BigInt(Number.MAX_SAFE_INTEGER)) {
    throw new Error(`Reference entry ${index} lies beyond ${Number.MAX_SAFE_INTEGER} bytes.`);
  }
  return { source, offset: Number(offset), length: Number(length) };
}

/** A source string starting with this asks for the reference-resolving store by name. */
export const JAZAV_PREFIX = "jazav://";
