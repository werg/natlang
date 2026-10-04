/** Record the effective runtime turn, retaining every wire attempt as separate evidence. */
export function recordingModelDriver({ createDriver, record, recordWire }) {
  return async (request, signal) => {
    const input = structuredClone(request), wireExchanges = [];
    // A callback belongs to one invocation, including its transport-level retries.
    const driver = createDriver(async exchange => {
      const snapshot = structuredClone(exchange);
      wireExchanges.push(snapshot);
      await recordWire?.(structuredClone(snapshot));
    });
    const turn = await driver(request, signal);
    await record({ recording_version: 'natlang.effective-model-turn/1', request: input,
      turn: structuredClone(turn), wireExchanges });
    return turn;
  };
}
