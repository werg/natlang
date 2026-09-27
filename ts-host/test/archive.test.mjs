import test from 'node:test';
import assert from 'node:assert/strict';
import { zipSync } from 'fflate';
import { openArchive } from '../dist/index.js';

const encoder = new TextEncoder();
function tar(path, content) {
  const body = encoder.encode(content), header = new Uint8Array(512);
  header.set(encoder.encode(path));
  header.set(encoder.encode(body.length.toString(8).padStart(11, '0') + '\0'), 124);
  header[156] = 48;
  const bytes = new Uint8Array(1024 + Math.ceil(body.length / 512) * 512);
  bytes.set(header); bytes.set(body, 512);
  return bytes;
}

test('ZIP and TAR archives open through the same folder API', async () => {
  const zip = openArchive(zipSync({ 'inbox/a.txt': encoder.encode('hello') }));
  const bundled = openArchive(tar('inbox/a.txt', 'hello'));
  for (const folder of [zip, bundled]) {
    assert.equal(await folder.file('inbox/a.txt').readText(), 'hello');
    assert.throws(() => folder.writeText('new.txt', 'no'), /read-only/);
  }
});

test('archives reject entries outside their root', () => {
  assert.throws(() => openArchive(zipSync({ '../secret.txt': encoder.encode('bad') })), /escapes folder/);
});
