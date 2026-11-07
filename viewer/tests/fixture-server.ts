import * as fs from "node:fs";
import * as http from "node:http";
import type { Server } from "node:http";
import * as path from "node:path";

const ROOT = path.resolve(__dirname, "..", "..", "fixtures", "local");

/** One request the server answered, for tests that count or inspect what a read costs. */
export interface ServedRequest {
  method: string;
  /** The path requested, decoded, without a query string. */
  path: string;
  /** The `Range` header as sent, if any. */
  range?: string;
  status: number;
}

/**
 * Serves `fixtures/local` over loopback, so a test can read a real OME-Zarr through
 * the same path a browser would -- FetchStore, HTTP, zarrita, the parsers -- without
 * depending on anything outside this machine.
 *
 * Loopback rather than a file:// store on purpose. The store is the part most likely
 * to break on a dependency upgrade, and reading bytes off disk would not exercise it.
 *
 * Range requests are honoured as S3, nginx and Caddy honour them: `206 Partial Content`
 * with a `Content-Range`, because a virtual image is read by byte range out of its
 * source files and a server that ignores `Range` would send the whole file instead.
 */
export async function startFixtureServer(): Promise<{
  url: string;
  requests: ServedRequest[];
  close: () => Promise<void>;
}> {
  const requests: ServedRequest[] = [];
  const server: Server = http.createServer((req, res) => {
    // Vizarr is served from a different origin to the data it reads, so the fixture
    // server answers as a public bucket does.
    res.setHeader("Access-Control-Allow-Origin", "*");
    res.setHeader("Access-Control-Allow-Methods", "GET, HEAD, OPTIONS");
    res.setHeader("Access-Control-Allow-Headers", "Range");
    res.setHeader("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges");
    res.setHeader("Accept-Ranges", "bytes");

    const requested = decodeURIComponent((req.url ?? "/").split("?")[0]);
    const range = req.headers.range;
    const served: ServedRequest = { method: req.method ?? "GET", path: requested, status: 0 };
    if (range) served.range = range;
    requests.push(served);
    const end = (status: number, headers: http.OutgoingHttpHeaders = {}, body?: Uint8Array) => {
      served.status = status;
      res.writeHead(status, headers).end(body);
    };

    const file = path.join(ROOT, requested);
    // Nothing outside the fixture directory is served, whatever the path says.
    if (!file.startsWith(`${ROOT}${path.sep}`)) {
      end(403);
      return;
    }
    fs.readFile(file, (error, contents) => {
      if (error) {
        // A missing key is how zarrita discovers that a node is absent -- a group
        // with no labels, a well with no image -- so this is an ordinary answer.
        end(404);
        return;
      }
      const total = contents.byteLength;
      const match = range ? /^bytes=(\d*)-(\d*)$/.exec(range) : null;
      if (match && (match[1] !== "" || match[2] !== "")) {
        // `bytes=a-b`, `bytes=a-` and the suffix form `bytes=-n`, as RFC 9110 defines them.
        const start = match[1] === "" ? Math.max(0, total - Number(match[2])) : Number(match[1]);
        const stop = match[1] === "" || match[2] === "" ? total - 1 : Math.min(Number(match[2]), total - 1);
        if (start > stop || start >= total) {
          end(416, { "Content-Range": `bytes */${total}` });
          return;
        }
        const body = contents.subarray(start, stop + 1);
        end(
          206,
          {
            "Content-Type": "application/octet-stream",
            "Content-Length": body.byteLength,
            "Content-Range": `bytes ${start}-${stop}/${total}`,
          },
          body,
        );
        return;
      }
      end(200, { "Content-Type": "application/octet-stream", "Content-Length": total }, contents);
    });
  });

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const address = server.address();
  if (address === null || typeof address === "string") {
    throw new Error("Expected the fixture server to be listening on a port");
  }
  return {
    url: `http://127.0.0.1:${address.port}`,
    requests,
    close: () => new Promise<void>((resolve) => server.close(() => resolve())),
  };
}
