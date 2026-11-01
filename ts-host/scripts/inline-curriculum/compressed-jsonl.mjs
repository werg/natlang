// Stream large replay journals directly to compressed storage with backpressure.
import { createWriteStream } from 'node:fs';
import { open } from 'node:fs/promises';
import { pipeline } from 'node:stream/promises';
import { constants, createZstdCompress } from 'node:zlib';

export function compressedJsonl(path) {
  const compressor = createZstdCompress({ params: { [constants.ZSTD_c_compressionLevel]: 6 } });
  const output = createWriteStream(path, { flags: 'wx' });
  const completion = pipeline(compressor, output);
  let failure;
  // Observe failures immediately, even while the caller is preparing the next row.
  completion.catch(error => { failure = error; });
  return {
    async write(text) {
      if (failure) throw failure;
      await new Promise((resolve, reject) => compressor.write(text, error => error ? reject(error) : resolve()));
      if (failure) throw failure;
    },
    async close() {
      compressor.end();
      await completion;
      const file = await open(path, 'r');
      try { await file.sync(); } finally { await file.close(); }
    },
    async abort(error) {
      compressor.destroy(error);
      output.destroy(error);
      await completion.catch(() => {});
    },
  };
}
