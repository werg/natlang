# OpenCode generation

Use the official OpenCode client for models restricted to its free tier. A
direct API request through the Pi provider is not an interchangeable transport.
Use the official SDK's supported permission controls to deny native IO tools
(wildcard deny plus inventory-derived deny, with the pure `invalid` rejection
handler allowed last). Audit complete histories and reject actual native IO.
Do not spoof the client's identity
or disguise another client as OpenCode.

## Credentials and availability

Pop already has `OPENCODE_API_KEY` in
`/home/werg/.config/natlang/opencode.env`, permission 0600. Do not put the key in
command arguments, version control, manifests, coordination notes, or logs.

The isolated client installation for the initial EXO investigation is
`runs/opencode-exo-client-20261007/client` (OpenCode CLI and SDK 1.18.35).
Refresh the client's model catalog with `opencode models opencode --refresh`
before requesting a newly listed model. This installation does not change the
repository's package dependencies.

EXO's model ID is `opencode/exo-free`. The initial authenticated official-client
probes on 2026-10-07 returned HTTP 503, `Endpoint is unavailable`. No EXO data
was generated or admitted by those probes. Disabling the client's built-in
tools instead triggered a free-tier gate; that is distinct from the upstream
availability failure.

An availability controller can retain isolated official-client probes:

```bash
python3 scripts/probe_opencode_model.py \
  --client "$PWD/runs/opencode-exo-client-20261007/client/node_modules/.bin/opencode" \
  --credentials /home/werg/.config/natlang/opencode.env \
  --out "$PWD/runs/opencode-exo-availability-FRESH" \
  --model opencode/exo-free
```

The output directory must be fresh. Default retry delay is five minutes,
doubling to thirty minutes with positive jitter. Each attempt has a 120-second
resource limit; timeout or operator stop kills its process group. A positive
receipt requires a successful CLI exit and the actual model reply `READY`.
Availability receipts never grant corpus admission. A ready probe exits; the
supervising agent must then exercise a real collector case and review its
result before assigning a larger generation queue.

## Structured-action bridge

The experimental SDK bridge serializes the ordered Natlang message history and
tool schemas into an official OpenCode `session.prompt` request asking for a
JSON text action envelope, validated locally against the declared shape. No
provider-enforced JSON Schema is claimed. Natlang executes the returned actions and owns the
next turn. OpenCode may itself run multiple internal assistant steps, so this
transport must inspect the entire isolated session and reject built-in tool
execution outside its structured-output mechanism.

Keep that transport provenance explicit. Its mapped tool-call IDs are host
correlation IDs, not provider-native IDs. Temperature, seed, and output limits
serialized into a prompt are not evidence that the provider enforced them.
Logprob decision scoring is unsupported. Buffered HTTP delivery is not
incremental generation. Preserve the original structured response and session
audit along with the mapped actions.

The loopback adapter's health endpoint describes adapter readiness only, not
provider availability. Begin with one real case and one request at a time;
expand concurrency after observing successful generation and provider limits.
Keep failed attempts, source pins, splits, and action reviews. Runtime success
and parent-case acceptance do not automatically admit every action to training.

`scripts/run_opencode_step5_collector.mjs` owns only the collector child it
spawns. On launcher `SIGINT` or `SIGTERM`, it forwards that signal to the child,
records `interrupted` plus the requested signal, and escalates to `SIGKILL` only
after 30 seconds if the child has not exited. Its outer timeout uses the same
bounded shutdown and records `timed_out`; a child spawn failure is recorded as
`spawn_error`. These lifecycle labels describe the collector process only and
do not prove that a provider request started or that the isolated OpenCode
bridge shut down. Preserve bridge capture logs and verify that bridge's own
lifecycle separately.

## Ownership

Run Pop's workers from `/home/werg/natlang`. The DGX agent manages DGX execution.
Synchronize the shared implementation through Git and publish approved data
with the existing corpus registry and immutable manifests. Neither provider
availability nor a copied artifact changes its source or quality admission.

## Verified history workaround (2026-10-07)

Official OpenCode 1.18.35 accepts a structured `format` prompt but its legacy
message-history endpoint then rejects the stored assistant `info.format`.
Removing the explicit history limit did not fix this: a real Bunny reproduction
failed with and without it. SDK v2 history routes are a separate session API.
The verified shared bridge requests strict JSON text without `format`, validates
it locally, and retains complete history and exact-final-message audit. Invalid
JSON is rejected without repair. See commit `7e3f6b87` and isolated real-turn
receipt `runs/opencode-bunny-json-text-history-20261007-v1/proof.json`.

CLI readiness alone is insufficient: initial real Bunny collector attempts also
revealed rejected native-tool protocol attempts named `invalid`. Keep these
failed receipts and investigate their handler/action evidence; do not broadly
allow OpenCode built-in tools or call failed transport attempts training data.
EXO availability probes still returned upstream 503 through attempt 7.
