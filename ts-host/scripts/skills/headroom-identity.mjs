import { createHash } from 'node:crypto';

export function headroomIdentity({executorId, pins, inputSha256, options}) {
  const allocation = {};
  for (const key of ['limit', 'concurrency', 'family-probe']) {
    const value = Number(options[key]);
    if (!Number.isSafeInteger(value) || value < (key === 'concurrency' ? 1 : 0))
      throw Error(`--${key} must be a ${key === 'concurrency' ? 'positive' : 'nonnegative'} integer`);
    allocation[key] = value;
  }
  if (!['train', 'validation', 'test', 'all'].includes(options.split)) throw Error('invalid --split');
  const band = options.band.split(',').map(Number);
  if (band.length !== 2 || !band.every(Number.isFinite) || !(band[0] >= 0 && band[1] <= 1 && band[0] < band[1]))
    throw Error('--band must be LOW,HIGH within [0,1]');
  return createHash('sha256').update(JSON.stringify({version:'natlang.episode-headroom-identity/3',
    executorId, pins, input_sha256:inputSha256, selection:{...allocation, split:options.split, band,
      databaseRoot:options['database-root'] ?? null, arenaRoot:options['arena-root'] ?? null},
    request:{temperature:0.2}})).digest('hex');
}

export function resumedHeadroomRows(body, identity) {
  const rows = new Map();
  for (const line of body.split('\n').filter(line => line.trim())) {
    const row = JSON.parse(line);
    if (row.schema !== 'natlang.episode-headroom/2' || row.screen !== identity)
      throw Error('Existing screen belongs to different input/runtime/executor/options; use a new output path');
    if (typeof row.episode !== 'string' || rows.has(row.episode)) throw Error('Duplicate or invalid resumed screen episode');
    rows.set(row.episode, row);
  }
  return rows;
}
