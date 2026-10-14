import { createReadStream } from 'node:fs';
import { createGunzip } from 'node:zlib';
import { StringDecoder } from 'node:string_decoder';
import { createHash } from 'node:crypto';

export async function* jsonlLines(path) {
  const source = createReadStream(path);
  const compressed = String(path).endsWith('.gz');
  const input = compressed ? createGunzip() : source;
  const forwardSourceError = error => { if (compressed) input.destroy(error); };
  if (compressed) {
    source.on('error', forwardSourceError);
    source.pipe(input);
  }
  const decoder = new StringDecoder('utf8');
  let fragments = [];
  try {
    for await (const chunk of input) {
      const decoded = decoder.write(chunk);
      let start = 0, at;
      while ((at = decoded.indexOf('\n', start)) >= 0) {
        fragments.push(decoded.slice(start, at));
        const line = fragments.join('');
        fragments = [];
        yield line.endsWith('\r') ? line.slice(0, -1) : line;
        start = at + 1;
      }
      if (start < decoded.length) fragments.push(decoded.slice(start));
    }
    const final = decoder.end();
    if (final) fragments.push(final);
    if (fragments.length) {
      const line = fragments.join('');
      yield line.endsWith('\r') ? line.slice(0, -1) : line;
    }
  } finally {
    if (compressed) source.off('error', forwardSourceError);
    if (!source.destroyed) source.destroy();
    if (input !== source && !input.destroyed) input.destroy();
  }
}

export async function* jsonlRows(path) {
  for await (const line of jsonlLines(path)) if (line.trim()) yield JSON.parse(line);
}

export async function fileDigest(path) {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}
