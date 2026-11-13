#!/usr/bin/env node
/**
 * The natlang command line: help, version and argument parsing here; each command's implementation is imported when
 * it runs (commands.ts and the per-area modules), so `natlang --version` and `natlang help` answer at once.
 */
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { CliError, EXIT, NATLANG_CLI_VERSION, output, parseArgs, usageError } from './args.js';
import { closestName } from '../compiler/suggest.js';
import { reportedFailure } from './progress.js';

export { NATLANG_CLI_VERSION } from './args.js';

const EXIT_CODES = 'Exit codes: 0 ok; 1 the program is wrong (diagnostics, failed examples, a failed call);\n2 usage error (options, arguments, paths); 3 the model or its runtime is unavailable.';

/** Rows of `natlang COMMAND ...` and a description, the descriptions aligned in one column across all `groups`. */
function tables(groups: [string, [string, string][]][]): string {
  const width = Math.max(...groups.flatMap(([, rows]) => rows.map(([usage]) => usage.length))) + 2;
  return groups.map(([title, rows]) => `${title ? `${title}:\n` : ''}${rows.map(([usage, description]) =>
    `  natlang ${usage.padEnd(width)}${description}`).join('\n')}`).join('\n\n');
}

function help(): string {
  return `natlang ${NATLANG_CLI_VERSION}

${tables([
    ['Start', [
      ['init [DIRECTORY]', 'Create a program with a sample input and examples.'],
      ['check [PROJECT]', 'Type-check a program without calling a model.'],
      ['call FUNCTION.nl [ARG...]', 'Call a natural-language function.'],
      ['test [DIRECTORY]', 'Run the examples in examples.json on the model.'],
      ['run [SOURCE] [-- ARGS]', 'Build and run a program.'],
      ['dev [SOURCE]', 'Run, and rerun after each edit.']]],
    ['Models', [
      ['setup', 'Install the local model runtime and model.'],
      ['models [PROVIDER]', 'List hosted model providers and their models.'],
      ['auth login|status|logout', 'Subscription sign-in for hosted model providers.'],
      ['doctor', 'Check the model, runtime and connection.'],
      ['dashboard', 'Open the local dashboard: models, programs, runs and traces.']]],
    ['Inspect', [
      ['traces list|show ...', 'The record of natlang calls.'],
      ['inspect SOURCE', 'Explain how a source resolves without running it.'],
      ['skills [install]', 'Show or install the natlang skills for Claude Code and Codex.']]]])}

${EXIT_CODES}
natlang help COMMAND shows a command's options; natlang help advanced lists the other commands.`;
}

const ADVANCED: [string, string][] = [
  ['build [PROJECT] [--warm]', 'Compile a program and write its lock (natlang help build).'],
  ['ask INSTRUCTION...', 'Ask over stdin, local files, or the folder.'],
  ['apply REDUCER DIR', 'Apply a directory reducer to a folder.'],
  ['apps [DIRECTORY]', 'Find runnable applications (natlang.json).'],
  ['tune [PROGRAM]', 'Tune a program\'s weights on its examples.'],
  ['setup --native SOURCE', 'Pin a trained native checkpoint and write its profile.'],
  ['serve [--profile NAME]', 'Keep a profile\'s native model server running.'],
  ['runtime status|install|build', 'Inspect, install or build the llama.cpp runtime.'],
  ['compilations list|show|why ...', 'Compiled cases and the calls they serve.'],
  ['specialize [--loop]', 'Compile recorded calls into guarded crisp cases.'],
  ['adapt inspect|export-source ...', 'Inspect trainable authored components.'],
  ['eval SUITE', 'Evaluate independent fixtures.'],
  ['optimize SUITE', 'Optimize instructions through the runtime.'],
  ['improve CASE.json --out DIR', 'Run the authored source-improvement program.'],
  ['packages', 'List installed distribution packages.'],
  ['package pack|verify|install ...', 'Pack, verify, or install a distribution archive.']];

const RUN_OPTIONS = `Options:
  --target NAME       Select one target when the manifest has several.
  --export NAME       Entry function for a TypeScript entry module (default main).
  --workspace DIR     Set the application's working directory.
  --state DIRECTORY   Override durable application state.
  --traces DIRECTORY  Override application trace storage.
  --profile NAME      Select a model profile.
  --provider ID       Use a hosted model provider (its newest Luna or Haiku model unless --model).
  --model ID          Override the profile's model ID.
  --store DIR         Package store for installed applications.
  --plain             Disable interactive terminal formatting.
  --no-color          Disable terminal color.
  --yes               Permit a required download of the model runtime and model.
  --progress MODE     auto (a status line on a terminal), line (a line every 15 s) or off;
                      also NATLANG_PROGRESS. A call that fails is always reported.`;

const RUNTIME = `Model runtime commands:
  natlang setup [--profile NAME] [--provider ID] [--model ID] [--yes] [--json]
  natlang setup --native SOURCE [--profile NAME] [--gguf q8_0|bf16|f32] [--escalate PROFILE|none] [--default] [--device D] [--port N] [--json]
  natlang setup --program DIR [--json]
  natlang serve [--profile NAME]
  natlang runtime status [--json]
  natlang runtime install [--yes] [--json]
  natlang runtime build [--cuda|--no-cuda] [--jobs N] [--json]

Setup reuses a compatible explicit, managed, or PATH llama-server. If none is
compatible, interactive use asks before installing the verified managed build and
downloading the model; without a terminal, pass --yes. Profiles live in
~/.config/natlang/config.json. Environment overrides are NATLANG_SERVER,
NATLANG_MODEL, NATLANG_MODEL_PATH, NATLANG_LLAMA_SERVER, NATLANG_RUNTIME_HOME,
NATLANG_PROVIDER, and NATLANG_API_KEY.

runtime build compiles natlang's llama.cpp fork (werg/llama.cpp-neuralese, at
the commit this natlang pins) into the runtime directory; managed sessions then
prefer it over the upstream build. It needs git, cmake and a C++ compiler, and
builds with CUDA when nvcc is on PATH.

setup --native pins a trained native checkpoint: SOURCE is a checkpoint file,
a JSON pointer to one (local_full_state, checkpoint or path, with sha256 and
global_step), or a name from the config's "nativeSources". The checkpoint is
copied, content-addressed, under the data directory (NATLANG_NATIVE_MODELS),
so a learner rewriting its file never changes what runs. The profile (default
name: the source name, else "native") starts the native server on its first
model call and stops it with the command; natlang serve keeps it running so
later commands start at once. --gguf also exports the weights for the managed
llama.cpp backend as profile NAME-gguf: faster, ordinary tokens only (the
Neuralese producer runs on the native server). --escalate PROFILE sets the fallback both profiles
hand off live invocations to (none removes it). Rerun setup to move to a newer checkpoint; runs
say when the source has one, and an existing NAME-gguf profile is exported again for it. The
server loads the pin's compact actor view (prepared once at setup: the actor weights without
optimizer state), a fraction of the full checkpoint's startup time and memory. A tuned program always runs on the checkpoint
its active release was tuned on, pinned on first use; setup --program DIR pins
it ahead of time. See docs/NATIVE_MODELS.md.`;

const CHECK = `Check or compile a project:
  natlang check [PROJECT] [--json] [--manifest] [--fusion] [--fusion-planner crisp|nl|shadow]
  natlang build [PROJECT] [--out DIR] [--target node|browser] [--warm] [--profile NAME] [--json]

PROJECT is a directory or tsconfig.json (default: .). Both commands write a
typed foo.d.nl.ts beside every foo.nl so editors and tsc see typed imports.
check --json prints { ok, diagnostics: [{ file, line, column, code, message, hint? }],
functions: [{ source, signature }] }; --manifest adds the full build manifest.

build also writes .natlang/build/natlang.lock.json: the model the program runs
with (natlang.json "model", else the selected profile; a stock model pinned by
revision and SHA-256), the llama.cpp build, the active tuning release and the
program's functions. --warm is the ahead-of-time tracing run: each natlang.json
"warmup" call ({ "function": "x.nl", "inputs": {...} }) runs with traces kept
under .natlang/build/warm/traces, and the prompt openings those calls send are
prefilled on the managed server and saved in the machine cache. A managed server
restores a saved opening into every slot before the first call that sends it
(and saves one it has not seen), so runs after a warm build start warm.
NATLANG_WARM_PREFIXES=off turns this off.

${EXIT_CODES}`;

const COMPILATIONS = `Compiled cases (guarded crisp TypeScript) for natural-language functions:
  natlang compilations list [--program DIR] [--all]
  natlang compilations show DEFINITION|ID    Cases, tiers, numbers, cases.ts and the report.
  natlang compilations why CALL              Which case served a call, or why none did.
  natlang compilations calls CASE [--role served|handed-off|training|held-out|shadow|audit]
  natlang compilations history DEFINITION
  natlang compilations export DEFINITION DIR Write the compilation and its evidence as files.
  natlang compilations disable|enable CASE|ID
  natlang compilations declines
  natlang specialize [--definition NAME] [--program DIR] [--loop] [--interval SECONDS] [--dry-run]
The runtime loads the current compilation of a function by itself: a new case starts in shadow and is
promoted by evidence; a case that fails hands the call back to the function.`;

const ADAPTATION = `Authored-component adaptation (docs/ADAPTATION.md):
  natlang adapt inspect [PROJECT] [--project DIR] [--json]
  natlang adapt export-source ARTIFACT [--project DIR] [--out FILE]
  natlang adapt revalidate ARTIFACT --suite SUITE [--mapping FILE] [--out FILE] [--dry-run|--preflight]
  natlang eval SUITE [--baseline | --artifact FILE] [--split train|validation|test] [--dry-run|--preflight] [--json]
  natlang optimize SUITE [--strategy gepa|reflection] [--out DIR] [--seed N] [--dry-run|--preflight] [--json]
  natlang optimize resume RUN --suite SUITE

A SUITE is a module exporting independent fixtures with expected outcomes; eval scores
the program (or an adaptation artifact) on one split, optimize searches instructions.`;

/** `natlang help TOPIC` and `natlang TOPIC --help`. Every command `natlang help` and `natlang help advanced` list has one. */
export const HELP_TOPICS: Record<string, string> = {
  advanced: `Other natlang commands:\n${tables([['', ADVANCED]])}\n\nnatlang help COMMAND shows a command's options.`,
  init: `Create a program without downloading or starting a model:
  natlang init [DIRECTORY]                   Default: my-program; existing files are never overwritten.

The program is answer.nl (a natural-language function), natlang.json (its run target,
whose input argument defaults to the sample text), examples.json (saved tests), a
README and AGENTS.md for coding agents. cd into it and run natlang run, or
natlang call answer.nl "text"; natlang dev reruns after edits; natlang test checks examples.json.`,
  skills: `Agent skills for writing natlang programs:
  natlang skills [list] [--json]              Where the natlang-authoring and natlang-integration skills are.
  natlang skills install [--claude] [--codex] [--dir DIR] [--json]

install copies both skill folders, replacing earlier copies, into ~/.claude/skills
(--claude), ~/.codex/skills (--codex) or DIR; without a choice, into each of those
agents that is set up on this machine. Restart or refresh the agent afterwards.`,
  tune: `Tune a program's weights on its own examples (docs/TUNING.md):
  natlang tune [PROGRAM] [--examples FILE]    Train on examples.json (or FILE), export, validate and activate a release.
  natlang tune [PROGRAM] --incremental|--full  Rebuild what changed, or refit every node.
  natlang tune [PROGRAM] --watch              Rebuild in the background on edits.
  natlang tune [PROGRAM] --report [--json]    The latest report, status and active release.
  natlang tune [PROGRAM] --rollback           Reactivate the previous release.
  natlang tune [PROGRAM] --evaluate development|confirmation|protected [--release DIGEST] [--with-teacher]
  natlang tune [PROGRAM] --rounds|--distill|--placement|--calibrate
  natlang tune --families PROGRAM...           Cross-program family fits.

PROGRAM defaults to the current directory. It trains on the program's examples.json (the file natlang test
reads); an entry without "expected" is an input the teacher labels. Tuning needs a native student profile
(natlang setup --native SOURCE); natlang doctor reports whether one is configured.`,
  dev: `Run and rerun after source changes:
  natlang dev [SOURCE] [RUN_OPTIONS] [-- APPLICATION_ARGS...]

Options are the same as natlang run. A change to a .nl, .nz, .ts, .mts or .js file,
natlang.json or another JSON input reruns the program (README, AGENTS.md and
examples.json do not). Ctrl+C stops the watcher and its current run.
Generated declarations, .natlang, dist and node_modules do not trigger reruns.`,
  test: `Run saved model examples:
  natlang test [DIRECTORY] [--json] [--profile NAME] [--provider ID] [--model ID] [--yes]

examples.json is an array of { name?, function: "answer.nl", inputs: {...}, expected: VALUE }.
Expected values are compared exactly, including object fields and array order.
These are real model calls; run natlang check for a model-free source check.
--json prints one report: { ok, passed, failed, seconds, results: [{ name, ok, call_id, expected,
received | error }] }; a build error prints { ok: false, diagnostics }, and a model that
cannot be reached prints { ok: false, error: { kind: "model-unavailable", message, next } }.
natlang traces show CALL_ID shows what a call did.

${EXIT_CODES}`,
  run: `Build and run an application:
  natlang run [SOURCE] [OPTIONS] [-- APPLICATION_ARGS...]

SOURCE is an application directory or natlang.json manifest (default: .), a
TypeScript entry module, NAME, or NAME@VERSION#TARGET of an installed package.
The project is compiled with natlang build into .natlang/build and the entry's
exported function (default main) receives the target context: args, io,
workspace, state and trace directories, the configured model, and a runtime.
A .nl entry prints its result as natlang call does.

${RUN_OPTIONS}

${EXIT_CODES}`,
  check: CHECK,
  build: CHECK,
  call: `Call a named natural-language function:
  natlang call NAME|FILE.nl [ARGUMENT...] [--json] [--trace DIR]
  natlang call NAME|FILE.nl [--inputs FILE] [--lines|--jsonl] [--trace DIR]
  model selection on either form: [--profile NAME] [--provider ID] [--model ID] [--yes]

NAME resolves in the current application's natlang.d context. ARGUMENTs fill the
parameters in order: text parameters as written, other types as JSON, for example
natlang call answer.nl "I was charged twice". A text result prints as text; --json
prints the result as JSON. Either way stderr ends with the call's ID
(natlang traces show CALL_ID shows what it did).
--inputs and piped stdin are JSON objects keyed by parameter name; for a function
with one text parameter, piped text that is not JSON is that text. --lines passes
each text line to a single-parameter function; --jsonl reads a JSON object per line.

${EXIT_CODES}`,
  apply: `Apply a directory reducer:
  natlang apply REDUCER DIR [--trace DIR] [--profile NAME] [--provider ID] [--model ID]

REDUCER is a name in the current natlang.d context or a .nl file. Successful
changes are saved to DIR after the reducer completes.`,
  ask: `Answer an instruction:
  natlang ask INSTRUCTION... [--returns TYPE] [--lines|--jsonl|--files GLOB] [--filter] [--jobs N]
  natlang ask INSTRUCTION... [--trace DIR] [--profile NAME] [--provider ID] [--model ID]

Without input, the instruction receives a read-only view of the current directory.
Piped stdin is passed as a whole, or one line/JSON record at a time. --files calls
once per matching file and prints path<TAB>result. --filter prints matching inputs.`,
  traces: `The machine's record of natlang calls (NATLANG_CALL_STORE, default ~/.local/share/natlang/calls):
  natlang traces status                      Store size, settings, pending shadow/audit jobs.
  natlang traces hot [--program DIR] [--by calls|tokens|wall_ms]
  natlang traces list [--definition NAME] [--executor agent|crisp|crisp-agent] [--outcome X] [--since ISO] [--all]
  natlang traces show CALL [--events] [--json]
  natlang traces occupancy [--definition NAME] [--limit N]   Batch occupancy: requests in flight, batch size, queue wait.
  natlang traces tiers [--definition NAME]                   Tiered engine: calls per tier, deopt rate, cost per tier.
  natlang traces export [--definition NAME] [--limit N]    One JSON record per line, values inlined.
  natlang traces pin|unpin CALL              Keep a call when the store evicts.
  natlang traces annotate CALL KIND VALUE    Attach feedback or a judgment to a call.
  natlang traces good CALL                   Tester verdict: annotate, and add the call's input and output as a root example
                                             in the program's tune.examples.jsonl (natlang tune reads it).
  natlang traces bad CALL [--expected JSON]  Annotate; with --expected also add the example (plain bad adds none).
  natlang traces fix CALL --expected JSON    Annotate and add the example with the right answer (replaces an earlier one).
    good|bad|fix options: --group G (else NATLANG_FEEDBACK_GROUP, else feedback-<day>-<host/user hash>: one tester's calls
    share a split side), --program DIR (program root; default the call's recorded root), --dry-run.
    Only root calls of a program .nl function become examples; others keep the annotation and say why.
  natlang traces config [KEY=VALUE...]       maxValueBytes, maxStoreBytes, specialization (off|shadow|on),
                                             auditRate, acceptanceBound, promotionComparisons, minCalls.
Every runtime records each call (exact inputs, output, service calls, eval code, cost) unless
NATLANG_CALL_STORE=off. A program's natlang.json can exclude values: "recording": { "exclude": [...] }.`,
  compilations: COMPILATIONS,
  specialize: COMPILATIONS,
  dashboard: `The local dashboard:
  natlang dashboard [--port N] [--host ADDRESS] [--root DIR] [--open]

Serves a web UI on loopback (default port 8780, the next free one when taken) for the
programs under DIR (default: .): configure model profiles, stock models and provider keys,
run programs and functions and watch their calls live, browse and annotate traces, ask the
model in a playground, and supervise builds, tuning and the specializer. The printed URL
carries the session's access token; --open also opens it in a browser.`,
  apps: `Discover applications:
  natlang apps [DIRECTORY] [--json]`,
  inspect: `Resolve a source without running it:
  natlang inspect SOURCE [--target NAME] [--store DIR] [--json]`,
  packages: `Distribution packages are optional deployment artifacts:
  natlang package pack MANIFEST_OR_DIRECTORY [--out FILE]
  natlang package verify ARCHIVE [--json]
  natlang package install ARCHIVE... [--store DIR] [--json]
  natlang packages [--store DIR] [--json]`,
  setup: RUNTIME,
  runtime: RUNTIME,
  serve: `Keep a native model server running:
  natlang serve [--profile NAME]

Starts the profile's native server (natlang setup --native SOURCE writes such a profile) and
holds it until Ctrl+C, so other natlang commands use it at once instead of starting their own.`,
  models: `List hosted model providers and models:
  natlang models [PROVIDER] [--json] [--refresh]

--json includes model capabilities and catalog defaults. Use one with --provider ID
(and --model ID) on run, call and test, or save it as a profile in natlang dashboard.`,
  auth: `Subscription sign-in for hosted model providers:
  natlang auth login PROVIDER
  natlang auth status [--json]
  natlang auth logout PROVIDER

login signs in with a provider subscription in the browser (providers that offer one).
API-key providers read their documented environment variable, such as ANTHROPIC_API_KEY.
natlang dashboard → Models connects providers (subscription sign-in or API key) from the browser.`,
  doctor: `Check the natlang installation and model configuration:
  natlang doctor [--profile NAME] [--provider ID] [--model ID] [--program DIR] [--store DIR] [--json]
Prints the model, the runtime, whether the model is reachable, where the cache and runtime live, and
the next action. --json prints the full report: the profile's native checkpoint (step, newer checkpoints,
actor view, whether its server is running), the escalation teacher, the tuning setup, and for a program
(--program, or the current directory when it has tuning state) its active release, the base it runs on,
examples waiting for natlang tune and when to retune.`,
  improve: `Run the authored source-improvement program:
  natlang improve CASE.json --out DIR --model MODEL [--provider ID] [--server URL] [--dry-run]
  natlang improve init [DIRECTORY] [--entry main.ts] [--export solve] [--out FILE]
  natlang improve resume|step DIR
  natlang improve inspect|export DIR [--out FILE]
  natlang improve adopt DIR --into CHECKOUT
  natlang improve rollback DIR

init writes a case for a program; the run proposes source changes, scores them on the
case's train/validation/test cases and keeps a journal in DIR (docs/IMPROVEMENT.md).`,
  adapt: ADAPTATION,
  eval: ADAPTATION,
  optimize: ADAPTATION,
};
HELP_TOPICS.package = HELP_TOPICS.packages!;

function topicHelp(topic: string): string {
  const found = HELP_TOPICS[topic];
  if (found) return found;
  const near = closestName(topic, Object.keys(HELP_TOPICS));
  throw usageError(`no help topic ${topic}${near ? ` (did you mean ${near}?)` : ''}; natlang help lists the commands`);
}

/** Commands with their own argument parsers; `--help` on them prints the topic here. */
const SEPARATE: Record<string, (argv: string[]) => Promise<number>> = {
  improve: async argv => (await import('./improvement.js')).improvementCommand(argv.slice(1)),
  adapt: async argv => (await import('./adaptation.js')).adaptationCommand(argv),
  eval: async argv => (await import('./adaptation.js')).adaptationCommand(argv),
  optimize: async argv => (await import('./adaptation.js')).adaptationCommand(argv),
  tune: async argv => (await import('./tune.js')).tuneCommand(argv.slice(1)),
  traces: async argv => (await import('./calls.js')).tracesCommand(argv.slice(1)),
  compilations: async argv => (await import('./calls.js')).compilationsCommand(argv.slice(1)),
  specialize: async argv => main((await import('./calls.js')).specializeArguments(argv.slice(1))),
};

/** Print an error once, as `natlang: MESSAGE`, and return its exit code. */
function failure(error: unknown): number {
  const message = error instanceof Error ? error.message : String(error);
  // A failure the progress display already reported (the call that failed) is not repeated.
  // The cause and the next action are one sentence: `natlang: CAUSE; NEXT`.
  const next = error instanceof CliError && error.next ? `; ${error.next[0]!.toLowerCase()}${error.next.slice(1).replace(/\.$/, '')}` : '';
  if (!reportedFailure(message)) process.stderr.write(`natlang: ${message}${next}\n`);
  return error instanceof CliError ? error.exitCode : EXIT.failed;
}

/** Run the command line `argv`; returns the exit code and prints any error (it never throws). */
export async function main(argv = process.argv.slice(2)): Promise<number> {
  try {
    const separate = SEPARATE[argv[0] ?? ''];
    if (separate && !argv.includes('--help') && !argv.includes('-h')) return await separate(argv);
    if (argv.includes('--version') && !argv.includes('--')) { output(NATLANG_CLI_VERSION, false); return 0; }
    const asksHelp = !argv.length || argv[0] === 'help' || argv.slice(0, argv.includes('--') ? argv.indexOf('--') : argv.length)
      .some(value => value === '--help' || value === '-h');
    if (asksHelp) {
      const topic = argv[0] === 'help' ? argv[1] : argv[0]?.startsWith('-') ? undefined : argv[0];
      output(topic ? topicHelp(topic) : help(), false); return 0;
    }
    const parsed = parseArgs(argv);
    const [command, ...words] = parsed.words;
    if (!command) throw usageError('natlang needs a command; natlang help lists them');
    if (!HELP_TOPICS[command] || command === 'advanced') {
      const near = closestName(command, Object.keys(HELP_TOPICS));
      throw usageError(`unknown command ${command}${near ? ` (did you mean ${near}?)` : ''}; natlang help lists the commands`);
    }
    return await (await import('./commands.js')).runCommand(command, words, parsed);
  } catch (error) { return failure(error); }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href)
  main().then(code => { process.exitCode = code; });
