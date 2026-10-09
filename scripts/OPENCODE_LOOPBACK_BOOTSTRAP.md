# OpenCode loopback bootstrap

`opencode-loopback-bwrap-launch.mjs` starts the official OpenCode SDK server and
the Natlang loopback Chat Completions adapter in a fresh scratch directory. It
binds both services to `127.0.0.1`; the adapter accepts one request at a time
by default. Set `--max-concurrency` to a value from 1 to 8 when the collector
will send that many simultaneous requests. The selected model defaults to
`exo-free`.

The launcher reads only `OPENCODE_API_KEY` from
`/home/werg/.config/natlang/opencode.env` after checking that file is mode
`0600`. The key is passed in the child environment; it is never accepted as a
command-line argument or written into the output directory. Bubblewrap exposes
the scripts and pinned SDK/client package tree read-only, gives the process
one writable run directory, hides the checkout and default home, and keeps
network access for the provider endpoint. Project config is disabled, and
user XDG config, plugins, and MCP servers are excluded. Pass absolute paths to
the installed SDK module and matching official `opencode` client binary:

```sh
node scripts/opencode-loopback-bwrap-launch.mjs \
  --sdk-module /path/to/node_modules/@opencode-ai/sdk/dist/v2/index.js \
  --client-bin /path/to/node_modules/opencode-linux-x64/bin/opencode \
  --out /path/to/new/empty-run-directory \
  --model exo-free \
  --max-concurrency 2
```

The launcher refuses an existing output directory. The bootstrap also accepts
a precreated empty output directory for the single-directory sandbox bind. It
writes an immutable
`bootstrap-config.json` containing sanitized paths, executable hashes, model,
loopback settings, and an explicit `provider_availability: not-probed` marker.
`lifecycle.json` tracks shutdown separately. The printed endpoint is loopback
only. The bootstrap changes into the fresh scratch directory before starting
the OpenCode server. The bwrap boundary keeps the rest of the checkout and
user home outside the server's filesystem view. Network access remains
available for the official provider transport; native network-capable tools
rely on pinned pre-effect permission asks and matching SDK event rejections.
History auditing detects unexpected tool use after the turn and is not itself
an execution barrier. `GET /health` checks the adapter process and does not
probe the provider.

## Step 5 Preview Free

Use the CLI-backed launcher `opencode-cli-loopback-bwrap-launch.mjs` with
`--model step-5-preview-free --variant low --tool-surface standard`. The launcher
takes the bare free model ID; the collector and generated OpenCode configuration
use `opencode/step-5-preview-free`. The bootstrap pins both its main and small model to this free
alias and supplies an explicit session title. Keep the ordinary built-in tool
schemas available under permission asks: removing them causes this provider's
free-tier route to reject requests. The bridge rejects native permission
requests and accepts actions through `natlang_action_bridge_submit_action`;
actual Natlang tools execute in the collector.

Before collecting, save the exact collector argument array as JSON and launch
through `scripts/run_opencode_step5_collector.mjs` with `--plan`,
`--bootstrap-config`, `--collector-argv-json`, and a fresh `--receipt` path.
The launcher checks the collector alias, main/small aliases, and reasoning
variant against the immutable bridge configuration. The plan declares the
outer collection resource bound. Fresh output paths preserve failed attempts
and their provider/runtime evidence. Retries use the collector's declared
delay and exponential backoff; a successful health check is not a model probe.

Current generation uses one concurrent request per bridge. This is a measured
starting configuration, not a provider-wide capacity limit. Raw accepted
results still need source/context/target review and explicit training admission.

The adapter returns buffered JSON even when the caller requests streaming, and
its response marks native provider tool calls and incremental streaming as
unsupported. The bridge asks for one strict JSON text object, parses it without
delimiter repair, and records that the provider did not enforce the response
schema. It can escape literal control characters inside JSON strings without
changing their decoded contents; malformed delimiters remain rejected. It
also accepts one whole-response JSON or unlabeled code fence around an otherwise
valid envelope. Literal backticks inside JSON string payloads are preserved;
extra prose, additional fences outside JSON strings, and undeclared tools fail
validation. Original response hashes and normalization kind remain recorded.
Provider retry events distinguish synthetic bridge status from the upstream
HTTP status, which the CLI may not expose. Preserve those failures as transport
evidence rather than incorrect model answers. The adapter preserves the parsed
action object and audited assistant usage in the raw
response. `SIGINT` and `SIGTERM` close the adapter
and official SDK server; during SDK startup the signal is also passed through
and startup is aborted before the adapter is created. Starting this script
does not create training data or grant training admission.
