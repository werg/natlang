export default function shiftCueInterval(startMs: number, endMs: number, oldSceneStartMs: number, newSceneStartMs: number): { startMs: number; endMs: number } {
  for (const value of [startMs, endMs, oldSceneStartMs, newSceneStartMs]) if (!Number.isSafeInteger(value) || value < 0) throw new Error("invalid_timeline_tick");
  if (endMs <= startMs) throw new Error("empty_cue_interval");
  const shift = newSceneStartMs - oldSceneStartMs;
  const start = startMs + shift; const end = endMs + shift;
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0) throw new Error("invalid_shifted_interval");
  return { startMs: start, endMs: end };
}
