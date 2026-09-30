import { createReadStream } from 'node:fs';
import { createGunzip } from 'node:zlib';
import { StringDecoder } from 'node:string_decoder';
import { createHash } from 'node:crypto';

export async function* jsonlRows(path) {
  const source = createReadStream(path);
  const compressed = String(path).endsWith('.gz');
  const input = compressed ? createGunzip() : source;
  const forwardSourceError = error => { if (compressed) input.destroy(error); };
  if (compressed) {
    source.on('error', forwardSourceError);
    source.pipe(input);
  }
  const decoder = new StringDecoder('utf8');
  let pending = '';
  const emitLine = line => {
    if (line.endsWith('\r')) line = line.slice(0, -1);
    return line.trim() ? JSON.parse(line) : undefined;
  };
  try {
    for await (const chunk of input) {
      pending += decoder.write(chunk);
      let at;
      while ((at = pending.indexOf('\n')) >= 0) {
        const line = pending.slice(0, at);
        pending = pending.slice(at + 1);
        const row = emitLine(line);
        if (row !== undefined) yield row;
      }
    }
    pending += decoder.end();
    if (pending.length) {
      const row = emitLine(pending);
      if (row !== undefined) yield row;
    }
  } finally {
    if (compressed) source.off('error', forwardSourceError);
    if (!source.destroyed) source.destroy();
    if (input !== source && !input.destroyed) input.destroy();
  }
}

export async function fileDigest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
