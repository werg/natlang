# Running natlang programs

How to run natlang programs and functions, what a compiled program carries, and the dashboard that supervises it all.

## The default model

A fresh install runs every natural-language call on the stock instruction model `LiquidAI/LFM3-1.7B` (Q8_0, 1.8 GB)
on natlang's managed llama.cpp server. Our own LFM3 checkpoints replace it once they are trained.

```sh
natlang setup --yes     # natlang's llama.cpp build (natlang-b12067) and the model, verified by SHA-256
natlang doctor          # what would run, and whether it is ready
```

Other choices are model profiles in `~/.config/natlang/config.json`, edited by hand or on the dashboard's Models page:

| Profile | Example |
|---|---|
| Another stock quantization | `{ "local": { "stock": "lfm3-1.7b-q4_k_m" } }` (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16) |
| A GGUF of your own | `{ "local": { "modelPath": "/models/x.gguf" } }` (served with the template it carries) |
| A GPU build of llama.cpp | `{ "local": { "server": "/opt/llama.cpp/build/bin/llama-server" } }` |
| An API provider | `{ "provider": "anthropic", "model": "claude-opus-5-5" }` (key: `natlang auth`, the dashboard, or the provider's variable) |
| An OpenAI-compatible server | `{ "endpoint": "http://127.0.0.1:8080", "model": "m", "apiKeyEnv": "MY_KEY" }` |
| Our trained checkpoints | `natlang setup --native SOURCE` ([native models](NATIVE_MODELS.md)) |

`--profile NAME`, `--provider ID --model ID`, `NATLANG_PROFILE`, `NATLANG_SERVER` and `NATLANG_PROVIDER` select one
for a single command.

A local profile's `request` refines the stock model's serving settings (greedy decoding, reasoning kept in the
history) field by field. The stock model reasons before every action, which is slow on a CPU (about 10 generated tokens
a second for Q4_K_M on four cores, and a turn often thinks for 1,500 tokens). `"request": { "thinking_budget_tokens":
512 }` caps each turn's reasoning (0 turns it off); the dashboard's profile editor sets it as **Thinking**. Unlimited
is how the model was evaluated, so a budget trades some accuracy for speed.

## Running

```sh
natlang run examples/triage                       # an application (natlang.json target)
natlang call examples/triage/classify.nl --inputs inputs.json
natlang ask "Summarize the input in one sentence." < notes.txt
```

While calls run, a terminal shows one status line: the function, its turn, prompt, cached and generated tokens, and
the elapsed time (`--progress line` prints a line every 15 seconds instead, `--progress off` nothing). A call that
fails is always reported, even when the program catches the error. A call whose model repeats the same failing action
four times in a row ends as `model_incomplete: repeated_failed_action` rather than running on without end.

## A compiled program carries its model and a warm start

A program states what it runs on in `natlang.json`:

```json
{
  "model": { "stock": "lfm3-1.7b-q8_0" },
  "warmup": [
    { "name": "outage ticket", "function": "classify.nl",
      "inputs": { "ticket": "Checkout fails with a 500 error for every customer.", "rubric": "billing: ...; technical: ..." } }
  ]
}
```

`model` takes a profile name, a stock model, a provider and model, or an endpoint, with `request`, `contextTokens` and
`parallel`; the command's own selection still wins. `natlang build` writes `.natlang/build/natlang.lock.json`: the
model (stock release pinned by repository revision and SHA-256, or the profile), the llama.cpp build, the active tuning
release and base checkpoint, and the program's functions.

`natlang build --warm` is the ahead-of-time tracing run. Each `warmup` call runs on that model, with its traces kept
under `.natlang/build/warm/traces` and in the call store; the lock records each call's outcome, time and tokens.

The warm start matters because every call opens with the same long system prompt and tool list (about 4,300 LFM3
tokens), and llama.cpp cannot reuse a hybrid model's cached prefix in the ordinary way: it restores LFM3's recurrent
and sliding-window state only from a checkpoint at least 256 tokens before the point where two prompts differ, and it
creates none there. Without help each call prefills the whole opening again, minutes on a CPU. The managed server
therefore prefills each distinct opening once, saves that slot state (whose checkpoints are early enough) in the machine
cache (`warm-prefixes/`, keyed by the exact opening, model file, llama.cpp build and slot layout, since a slot state
restores only into a server with the same slots; the 24 most recent are kept), and
restores it into every slot before the first call that sends it. Later calls in a warm slot keep reusing it, and later
runs start warm. The warm build fills this cache ahead of time and lists the openings in the lock. Measured on four CPU
cores with Q4_K_M, a first call reused 2,923 of its 3,439 opening tokens after a 0.07 s restore. `NATLANG_WARM_PREFIXES=off`
turns it off; states are not used while tuned LoRA adapters or native states are loaded.

## One loaded model for every run

The managed server sizes its slots from the model's measured context memory (LFM3 1.7B: about 12.5 KiB per token, so
the default 4 GiB budget holds 8 slots of 32,768 tokens). The dashboard's **Keep the model loaded** starts one server
and publishes it in the state directory (`model-server.json`); every natlang command on the machine whose model file and
llama.cpp build match uses it instead of starting its own, whether it was started from the dashboard or a terminal: no
load time, the server's warm slots, and one copy of the model in memory. Runs warm openings on it only while all its
slots are idle. `NATLANG_SHARED_SERVER=off` makes a command start its own server.

## llama.cpp

The managed runtime is a release of natlang's llama.cpp fork
[werg/llama.cpp-neuralese](https://github.com/werg/llama.cpp-neuralese), currently natlang-b12067 (llama.cpp 0.8.0):
prebuilt CPU builds for Linux and Windows (x64, arm64) and Metal builds for macOS, checked by SHA-256. The fork adds
decision scoring, guided generation (whose reasoning streams live), native states, RL capture and backend sampling; an
upstream llama.cpp serves the same programs without those. Its `natlang-runtime` workflow publishes a release and
`node scripts/update_llama_runtime.mjs --fork natlang-bNNNN` pins it. For a GPU, or a fork commit newer than the
release, `natlang runtime build` compiles the commit this natlang pins (`ts-host/src/llama-fork-source.ts`; it needs
git, cmake and a C++ compiler, and uses CUDA when `nvcc` is on PATH) into the runtime directory, and managed sessions
prefer that build from then on; the System page has the same button and says which build runs. `NATLANG_LLAMA_SERVER`
(or a profile's `local.server`) still selects any other binary.

## The dashboard

```sh
natlang dashboard --open          # http://127.0.0.1:8780, for the programs under the current directory
natlang dashboard --root ~/work --port 9000
```

It binds to loopback; the printed URL carries a session token that every request must present. Pages:

- **Overview**: the selected model, runtime, call store and machine; the kept-loaded model; recent runs; the busiest
  functions.
- **Programs**: every `natlang.json` under the root. Run a target with arguments and a profile, call any `.nl` function
  from a form generated from its signature, build or build warm, and see the lock and warm-up results.
- **Runs**: every run, call, ask, build, setup, tuning and specializer job, live: the tree of calls, each turn's
  reasoning, code and failures, token counts, the program's output with an input box for interactive programs, and the
  log. Jobs are ordinary CLI processes; their records stay under the state directory (`dashboard/jobs/`).
- **Traces**: the machine's call records with filters; a call's inputs, output, evals, service calls, child calls,
  annotations and full event trace; mark it good, bad, or fix it with the expected result (adds a tuning example).
- **Playground**: one `natlang ask` with an instruction, input, result type and profile.
- **Models**: profiles (local stock model, API provider with its key, OpenAI-compatible server), the default profile,
  stock model downloads with progress, and a one-click test of each profile.
- **System**: the llama.cpp runtime (upstream or natlang's fork) and its installation, call-store specialization mode,
  the saved warm prompt openings, paths.
