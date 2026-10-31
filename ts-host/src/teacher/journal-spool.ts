import { createHash, randomUUID } from 'node:crypto';
import { mkdir, open, readFile, readdir, rename, rm, stat, statfs, copyFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { setTimeout as delay } from 'node:timers/promises';

export const JOURNAL_SPOOL_CASE_LIMIT = 256 * 1024 * 1024;
export const JOURNAL_SPOOL_ACTIVE_CASE_LIMIT = 6;
export const JOURNAL_SPOOL_TOTAL_LIMIT = JOURNAL_SPOOL_CASE_LIMIT * JOURNAL_SPOOL_ACTIVE_CASE_LIMIT;
export const JOURNAL_SPOOL_FREE_FLOOR = 40 * 1024 * 1024 * 1024;

function hashFile(path: string): Promise<string> {
  return new Promise((resolveHash, reject) => {
    import('node:fs').then(({ createReadStream }) => {
      const hash = createHash('sha256');
      const stream = createReadStream(path);
      stream.on('data', chunk => hash.update(chunk));
      stream.on('error', reject);
      stream.on('end', () => resolveHash(hash.digest('hex')));
    }).catch(reject);
  });
}

async function syncDirectory(path: string): Promise<void> {
  const handle = await open(path, 'r');
  try { await handle.sync(); } finally { await handle.close(); }
}

function pidAlive(pid: number): boolean {
  if (!Number.isInteger(pid) || pid < 1) return false;
  try { process.kill(pid, 0); return true; }
  catch (error) { return !!error && typeof error === 'object' && 'code' in error && error.code === 'EPERM'; }
}

async function withRegistryLock<T>(root: string, body: () => Promise<T>): Promise<T> {
  const lock = join(root, '.journal-spool-lock');
  await mkdir(root, { recursive: true });
  const token = `${process.pid}:${randomUUID()}`;
  const until = Date.now() + 30_000;
  while (true) {
    try {
      const handle = await open(lock, 'wx');
      await handle.writeFile(token);
      await handle.sync();
      await handle.close();
      try { return await body(); }
      finally {
        if (await readFile(lock, 'utf8').catch(() => '') === token) await rm(lock, { force: true });
      }
    } catch (error) {
      if (!error || typeof error !== 'object' || !('code' in error) || error.code !== 'EEXIST') throw error;
      const owner = await readFile(lock, 'utf8').catch(() => '');
      const lockPid = Number(owner.split(':', 1)[0]);
      if (owner && !pidAlive(lockPid)) { await rm(lock, { force: true }); continue; }
      if (!owner) {
        const info = await stat(lock).catch(() => undefined);
        if (!info || Date.now() - info.mtimeMs > 1000) { await rm(lock, { force: true }); continue; }
      }
      if (Date.now() >= until) throw new Error('timed out acquiring journal-spool registry lock');
      await delay(10);
    }
  }
}

async function treeBytes(path: string): Promise<number> {
  let total = 0;
  const walk = async (dir: string): Promise<void> => {
    for (const entry of await readdir(dir, { withFileTypes: true }).catch(() => [])) {
      const child = join(dir, entry.name);
      if (entry.isDirectory()) await walk(child);
      else if (entry.isFile()) total += (await stat(child)).size;
    }
  };
  await walk(path);
  return total;
}

export type JournalSpool = {
  directory: string;
  caseIdentitySha256: string;
  assertCapacity(additionalBytes: number): Promise<void>;
  publish(files: Array<{ source: string; destination: string; sha256: string }>): Promise<void>;
};

/** A crash-recoverable, bounded local journal. Its lease is renewed on restart and freed only after verified publication. */
export async function acquireJournalSpool(rootInput: string, caseIdentity: string): Promise<JournalSpool> {
  const root = resolve(rootInput), key = createHash('sha256').update(caseIdentity).digest('hex');
  const leases = join(root, '.leases'), casesRoot = join(root, 'cases'), directory = join(casesRoot, key);
  await mkdir(leases, { recursive: true });
  await mkdir(casesRoot, { recursive: true });
  await withRegistryLock(root, async () => {
    const entries = await readdir(leases);
    for (const name of entries) {
      const path = join(leases, name);
      const lease = JSON.parse(await readFile(path, 'utf8').catch(() => '{}')) as { key?: string; pid?: number };
      if (lease.key !== key && !pidAlive(Number(lease.pid))) await rm(path, { force: true });
    }
    const current = await readdir(leases);
    if (!current.includes(`${key}.json`) && current.length >= JOURNAL_SPOOL_ACTIVE_CASE_LIMIT)
      throw new Error(`journal spool has ${JOURNAL_SPOOL_ACTIVE_CASE_LIMIT} active-case credits in use`);
    const retainedCases = await readdir(join(root, 'cases'), { withFileTypes: true }).catch(() => []);
    if (!retainedCases.some(entry => entry.name === key) && retainedCases.filter(entry => entry.isDirectory()).length >= JOURNAL_SPOOL_ACTIVE_CASE_LIMIT)
      throw new Error(`journal spool retains ${JOURNAL_SPOOL_ACTIVE_CASE_LIMIT} case directories; recover or publish an interrupted case before admitting another`);
    const fs = await statfs(root);
    const availableBytes = Number(fs.bavail) * Number(fs.bsize);
    const reserved = (current.includes(`${key}.json`) ? current.length : current.length + 1) * JOURNAL_SPOOL_CASE_LIMIT;
    if (availableBytes - reserved < JOURNAL_SPOOL_FREE_FLOOR)
      throw new Error(`journal spool reservation would cross the 40 GiB internal free-space floor (${availableBytes} available, ${reserved} reserved)`);
    await mkdir(directory, { recursive: true });
    const leasePath = join(leases, `${key}.json`);
    const lease = { schema: 'natlang.teacher_journal_spool_lease/1', key, case_identity_sha256: key,
      pid: process.pid, updated_at: new Date().toISOString(), per_case_limit_bytes: JOURNAL_SPOOL_CASE_LIMIT };
    const temp = `${leasePath}.${randomUUID()}.tmp`;
    await (await open(temp, 'wx')).close();
    const handle = await open(temp, 'w'); await handle.writeFile(JSON.stringify(lease) + '\n'); await handle.sync(); await handle.close();
    await rename(temp, leasePath);
  });
  return {
    directory,
    caseIdentitySha256: key,
    assertCapacity: async additionalBytes => {
      if (!Number.isSafeInteger(additionalBytes) || additionalBytes < 0) throw new RangeError('journal spool byte reservation must be nonnegative');
      const used = await treeBytes(directory);
      const aggregate = await treeBytes(join(root, 'cases'));
      if (used + additionalBytes > JOURNAL_SPOOL_CASE_LIMIT || aggregate + additionalBytes > JOURNAL_SPOOL_TOTAL_LIMIT)
        throw new Error(`journal spool case cap exceeded (${used}+${additionalBytes}>${JOURNAL_SPOOL_CASE_LIMIT})`);
    },
    publish: async files => {
      for (const file of files) {
        const actual = await hashFile(file.source);
        if (actual !== file.sha256) throw new Error(`journal spool source changed before publication: ${file.source}`);
        await mkdir(dirname(file.destination), { recursive: true });
        const temp = `${file.destination}.${randomUUID()}.spool-tmp`;
        await copyFile(file.source, temp);
        const copyHash = await hashFile(temp);
        if (copyHash !== file.sha256) { await rm(temp, { force: true }); throw new Error(`journal spool publication digest mismatch: ${file.destination}`); }
        const handle = await open(temp, 'r'); await handle.sync(); await handle.close();
        await rename(temp, file.destination);
        await syncDirectory(dirname(file.destination));
      }
      await withRegistryLock(root, async () => {
        await rm(directory, { recursive: true, force: true });
        await rm(join(leases, `${key}.json`), { force: true });
      });
    },
  };
}
