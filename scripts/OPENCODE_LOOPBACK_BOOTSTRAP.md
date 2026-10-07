# OpenCode loopback bootstrap

`opencode-loopback-bootstrap.mjs` starts the official OpenCode SDK server and
the Natlang loopback Chat Completions adapter in a fresh scratch directory. It
binds both services to `127.0.0.1`; the adapter accepts one request at a time
by default. Set `--max-concurrency` to a value from 1 to 8 when the collector
will send that many simultaneous requests. The selected model defaults to
`exo-free`.

Provide the API key through the process environment using the existing secure
credential setup. The key is never accepted as a command-line argument and is
never written into the output directory. Pass absolute paths to the installed
SDK module and matching official `opencode` client binary explicitly:

```sh
node scripts/opencode-loopback-bootstrap.mjs \
  --sdk-module /path/to/node_modules/@opencode-ai/sdk/dist/v2/index.js \
  --client-bin /path/to/node_modules/opencode-linux-x64/bin/opencode \
  --out /path/to/new/empty-run-directory \
  --model exo-free \
  --max-concurrency 2
```

The script refuses an existing output directory. It writes an immutable
`bootstrap-config.json` containing sanitized paths, executable hashes, model,
loopback settings, and an explicit `provider_availability: not-probed` marker.
`lifecycle.json` tracks shutdown separately. The printed endpoint is loopback
only. The bootstrap changes into the fresh scratch directory before starting
the OpenCode server, so project files in the checkout are not loaded as its
working context. `GET /health` checks the adapter process and does not probe
the provider.

The adapter returns buffered JSON even when the caller requests streaming, and
its response marks native provider tool calls and incremental streaming as
unsupported. The bridge asks for one strict JSON text object, parses it without
repair, and records that the provider did not enforce the response schema. It
preserves the parsed action object and audited assistant usage in the raw
response. `SIGINT` and `SIGTERM` close the adapter
and official SDK server; during SDK startup the signal is also passed through
and startup is aborted before the adapter is created. Starting this script
does not create training data or grant training admission.
