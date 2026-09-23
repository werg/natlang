# Runtime parity (retired)

The Python runtime was removed in the TypeScript-native refactor, so the
cross-runtime parity gate, its change-detection script, and the differential
tests were removed with it. Node and browser now share one TypeScript
implementation; their parity is covered by `ts-host/test/browser.test.mjs`, the
real-browser smokes (`npm run test:browser`, `npm run test:playground`), and the
shared conformance programs (`npm run test:conformance`).
