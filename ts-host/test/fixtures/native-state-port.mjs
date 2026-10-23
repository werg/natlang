import { nativeStateFixture } from './native-state.mjs';

/** Test-only writer that archives a real native-state fixture with the requested exact text and type. */
export function createNativeStateFixturePort(store, dialect = 'native-state/fixture') {
  const prefix = 'native-state/';
  if (!dialect.startsWith(prefix) || !dialect.slice(prefix.length)) throw new TypeError('fixture port dialect must name a native-state architecture');
  const architecture = dialect.slice(prefix.length);
  return {
    dialect,
    async write(text, options = {}) {
      const fixture = nativeStateFixture();
      const artifact = await store.putNativeState({
        ...fixture,
        metadata: { ...fixture.metadata, architecture, text },
      });
      const meta = await store.meta(artifact.id);
      return { ...meta, ...(options.type ? { type: options.type } : {}),
        ...(options.truncated ? { truncated: true } : {}),
        producer: { ...(options.producer ?? {}), ...(options.truncated ? { truncated: true } : {}) } };
    },
  };
}
