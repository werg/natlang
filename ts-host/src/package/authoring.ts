/** Project scaffolding and saved input/output examples. No model calls are needed to create a program. */
import { existsSync, mkdirSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';

export const sampleInput = 'I was charged twice for my subscription.';
const starter = {
  description: 'Classify a customer message as billing, technical or other.',
  parameter: 'message',
  instruction: 'Classify message as billing (charges, invoices or payments), technical (errors or bugs), or other. Return only the matching label.',
  returns: '"billing" | "technical" | "other"',
};
// A YAML single-quoted scalar: type text keeps its double quotes readable (`'"a" | "b"'`).
const quoted = (text: string) => `'${text.replace(/'/g, "''")}'`;

/**
 * A program's `answer.nl`. The starter is a finite classification, so it declares `readout: decision`: the model reasons
 * and scores each label instead of running a tool loop.
 * Instructions supplied by the caller (the dashboard Playground) take one `input` and return `returns`.
 */
export const answerSource = (custom?: { instruction: string; returns: string }) => {
  const { description, parameter, instruction, returns } = custom ?
    { description: 'Answer using the program instructions.', parameter: 'input', ...custom } : starter;
  return `---\ndescription: ${description}\nargs:\n  ${parameter}: string\nreturns: ${quoted(returns)}\n${custom ? '' : 'readout: decision\n'}---\n${instruction}\n`;
};

/** The programs `natlang init` writes: `inbox` (the default) or the one-function `classifier`. */
export const TEMPLATES = ['inbox', 'classifier'] as const;
export type Template = typeof TEMPLATES[number];

export function createProgram(directory: string, options: { instruction?: string; input?: string; returns?: string; profile?: string; template?: Template } = {}): string {
  const root = resolve(directory);
  if (existsSync(root) && (!statSync(root).isDirectory() || readdirSync(root).length)) throw new Error(`${root} is not empty. Choose a new directory; existing files are never overwritten.`);
  const name = basename(root).toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '') || 'my-program';
  const custom = options.instruction?.trim() ? { instruction: options.instruction.trim(), returns: options.returns?.trim() || 'string' } : undefined;
  const template = custom ? 'classifier' : options.template ?? 'inbox';
  if (!TEMPLATES.includes(template)) throw new Error(`There is no ${template} template; the templates are ${TEMPLATES.join(' and ')}.`);
  const files = template === 'inbox' ? inboxProgram(name, options.profile) : classifierProgram(name, { ...options, custom });
  mkdirSync(root, { recursive: true });
  for (const [name, content] of Object.entries(files)) writeFileSync(join(root, name), content, { flag: 'wx' });
  return root;
}

const json = (value: unknown) => JSON.stringify(value, null, 2) + '\n';
const fence = '\x60\x60\x60';
const tick = '\x60';

/**
 * The default program (plans/semantic-fp/04_DOCS_EXAMPLES_AND_AUTHORING.md section 2): an inbox planner whose types,
 * data and exact steps are TypeScript and whose judgments are typed natural-language functions. `route.nl` is a
 * decision readout (reason, then score; calibrated probabilities through `decide`), and the
 * reply draft is an inline `nl` with a checked property type, called only for the messages that need a reply.
 */
function inboxProgram(name: string, profile?: string): Record<string, string> {
  // The bundled messages need no reply, so the first run is three scored decisions: about a minute on a CPU. A
  // drafted reply is a generative call that reasons for minutes on a CPU (measured 2026-11-23, plans/neuralese/
  // DECISIONS.md "natlang init writes the inbox planner"); the README shows how to run one with your own message.
  const messages = [
    { from: 'lee@example.com', subject: 'New designer', body: 'Hi team, Maria joins us on Monday as our new designer.' },
    { from: 'deals@shop.example', subject: '50% off this weekend only', body: 'Our biggest sale of the year starts now. Click to shop.' },
    { from: 'finance@example.com', subject: 'Q3 report published', body: 'The Q3 report is now on the wiki.' },
  ];
  const routed: [string, string, { from: string; subject: string; body: string }][] = [
    ['a question needs a reply', 'reply', { from: 'sam@example.com', subject: 'Meeting time', body: 'Does Tuesday at 10 still work for our review?' }],
    ['an announcement is fyi', 'fyi', { from: 'hr@example.com', subject: 'New holiday calendar', body: 'The 2027 holiday calendar is now on the intranet.' }],
    ['a newsletter is ignored', 'ignore', { from: 'news@vendor.example', subject: 'Weekly digest', body: 'Top stories this week from our blog. Unsubscribe at any time.' }],
  ];
  return {
    'types.ts': TYPES_TS,
    'route.nl': ROUTE_NL,
    'main.ts': MAIN_TS,
    'messages.json': json(messages),
    'examples.json': json(routed.map(([label, expected, message]) => ({ name: label, function: 'route.nl', inputs: { message }, expected }))),
    'natlang.json': json({ schema: 'natlang.package/v2', name, version: '0.1.0', description: 'Plan an inbox: route every message, draft the replies it needs.',
      ...(profile ? { model: { profile } } : {}),
      include: ['examples.json', 'main.ts', 'messages.json', 'natlang.d', 'package.json', 'README.md', 'route', 'route.nl', 'tsconfig.json', 'types.ts'],
      targets: { main: { entry: 'main.ts', description: 'Plan the bundled messages.json, or messages you give as JSON.',
        arguments: [{ name: 'messages', type: 'Message[]', description: 'Messages to plan, as JSON; leave blank for messages.json.' }] } },
      warmup: [{ name: 'one message', function: 'route.nl', inputs: { message: messages[0] } }] }),
    'package.json': json({ name, private: true, type: 'module' }),
    'tsconfig.json': json({ compilerOptions: { target: 'ES2023', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true, skipLibCheck: true,
      noEmit: true, types: ['node'] }, include: ['*.ts'] }),
    '.gitignore': '.natlang/\nnode_modules/\n*.d.nl.ts\n',
    'README.md': inboxReadme(name),
    'AGENTS.md': INBOX_AGENTS,
    'CLAUDE.md': '@AGENTS.md\n',
  };
}

const TYPES_TS = `/** One incoming message. The body is outside text: the runtime fences it and never follows instructions in it. */
export type Message = { from: string; subject: string; body: Untrusted<string> };
/** What the message needs from its reader. */
export type Route = 'reply' | 'fyi' | 'ignore';
/** A drafted reply: short, polite, and promising nothing the message does not ask for. */
export type Reply = Is<string, 'a polite reply of at most three sentences that makes no commitment the message does not ask for'>;
export type Plan = { from: string; subject: string; route: Route; confidence: number; reply?: Reply };
`;

const ROUTE_NL = `---
description: Decide what a message needs from its reader.
args:
  message: Message
returns: Route
readout: decision
---
reply when the sender asks the reader a question or asks them to do something; fyi when a person or team tells the
reader something worth knowing and asks nothing; ignore when it is advertising, a newsletter or an automated message
the reader does not need.
`;

const MAIN_TS = `import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { decide, nl, untrusted, type TargetContext } from '@natlang/node';
import route from './route.nl';
import type { Message, Plan, Reply } from './types.js';

/** Plan the reader's inbox: route every message, draft replies for the ones that need one. */
export async function plan(messages: Message[]): Promise<Plan[]> {
  // Independent judgments run together; each is a calibrated decision on the local model.
  const decisions = await Promise.all(messages.map(message => decide(route, message)));
  const draft = nl<Reply>${tick}Draft a reply to message for its reader.${tick};
  return Promise.all(messages.map(async (message, i) => {
    const { value, confidence } = decisions[i]!;
    const planned: Plan = { from: message.from, subject: message.subject, route: value, confidence };
    if (value === 'reply') planned.reply = await draft(message);
    return planned;
  }));
}

/** ${tick}natlang run${tick}: plan the messages given as JSON, or the bundled messages.json. */
export async function main(context: TargetContext): Promise<number> {
  const given = context.arguments.messages as Message[] | undefined;
  const raw = given ?? JSON.parse(readFileSync(join(context.package?.root ?? context.workspace, 'messages.json'), 'utf8')) as Message[];
  const messages = raw.map(message => ({ ...message, body: untrusted(message.body, ${tick}message from \${message.from}${tick}) }));
  context.io.output.write(JSON.stringify(await context.runtime.run(() => plan(messages)), null, 2) + '\\n');
  return 0;
}
`;

const inboxReadme = (name: string) => `# ${name}

${fence}sh
natlang run                          # plan the bundled messages.json (three decisions)
natlang run . -- '[{"from":"a@b","subject":"Hi","body":"Are you free at 3?"}]'   # one that drafts a reply
natlang call route.nl '{"from":"a@b","subject":"Invoice","body":"Can you resend it?"}'
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
${fence}

**types.ts** is the data: a \`Message\` whose body is \`Untrusted<string>\` (the model reads it as data and never follows
it), the \`Route\` a message can take, and a \`Reply\` whose property the runtime checks on every draft.

**route.nl** is a natural-language function: a typed signature and the instructions a model carries out. Its
\`readout: decision\` makes the model reason, then score the three routes, and \`decide\` returns their probabilities.

**main.ts** is the program: it routes every message at once, drafts a reply with an inline \`nl<Reply>\` function only
where one is needed, and keeps each decision's confidence. \`main\` is what \`natlang run\` calls.

**examples.json** saves inputs with their expected results; \`natlang test\` runs them on the real model.
**messages.json** is the sample input. Callable folders hold what a function may call, created when needed: **route/**
for \`route.nl\`, **natlang.d/** for the inline \`nl\` in main.ts (\`.ts\` for exact steps, \`.nl\` for judgments).

\`natlang check\` checks the program without calling a model. Every call prints its call ID on stderr: \`natlang traces
show CALL_ID\` shows the steps the model took. The first run prepares the selected model: the local one is a one-time
download (1.8 GB), or choose a hosted model on the dashboard's Models page. AGENTS.md is the guide for coding agents.
`;

const INBOX_AGENTS = `# Working on this natlang program

This program automates a piece of judgment work with a small local model. TypeScript holds the types, the data and
every exact step; each judgment is a typed natural-language function the model runs. Read \`main.ts\` first: it is the
whole program in one screen.

## Files

- \`types.ts\`: the data. \`Untrusted<string>\` marks outside text (the model reads it as data and never follows it);
  \`Is<string, "...">\` is a property the runtime checks on every value and repairs when it fails.
- \`route.nl\`: a named natural-language function. YAML front matter declares its typed signature; the body says what
  to decide. \`readout: decision\` makes the model reason, then score each allowed value: calibrated
  (\`decide(route, message)\` gives you the probabilities).
- \`main.ts\`: TypeScript that maps \`route\` over the messages, and an inline function, \`\`nl<Reply>\`Draft a reply to
  message for its reader.\` \`\`, called only where a reply is needed. An inline \`nl\` takes its call's arguments under
  the names the call writes (\`draft(message)\` gives it \`message\`) and captures the other variables its instructions
  name. \`main\` is what \`natlang run\` calls.
- \`natlang.json\`: the package. \`targets.main\` runs main.ts's \`main\`; its \`messages\` argument has a type from
  types.ts, checked before the run. \`include\` lists what the package holds, folders included before they exist.
- \`examples.json\`: cases with expected results; \`natlang test\` runs them on the model.
- Callable folders (create them when needed): \`route/\` holds what \`route.nl\` may call, \`natlang.d/\` what the
  inline \`nl\` in main.ts may call; \`.ts\` for exact steps, \`.nl\` for judgments.
- Generated (do not edit): \`*.d.nl.ts\` and \`.natlang/\`.

## The loop

1. \`natlang check --json\`: types, signatures and call sites, without a model. Fix every diagnostic first; each one
   says what to write instead.
2. \`natlang call route.nl '{"from":"a@b","subject":"Invoice","body":"Can you resend it?"}' --json\`: one real call.
   \`natlang run\` runs the whole program on messages.json, whose messages need no reply; \`natlang run . -- '[...]'\`
   plans your own messages as JSON, and drafts a reply for each one that needs it.
3. \`natlang test --json\`: every case in examples.json. When you change behaviour, add a case first.
4. \`natlang traces show CALL_ID\`: every step the model took (each call prints its ID on stderr).
   \`natlang traces fix CALL_ID --expected JSON\` turns a wrong result into a saved case.

Exit codes: 0 ok; 1 the program is wrong; 2 usage error; 3 the model is unavailable.

## Growing the program

- A new judgment is a new typed function: a \`.nl\` file when it is reused or has its own cases, an inline \`nl<T>\`
  when it belongs to one place in the code. Give it the narrowest result type that fits: a literal union with
  \`readout: decision\` when the answer is one of a few values, a record when it has parts.
- An exact step (counting, sorting, parsing, joining records, arithmetic) is a TypeScript function; a \`.nl\` function
  that needs it calls it from its callable folder.
- A property of a result goes in its type (\`Is<T, "...">\`), and outside text in \`Untrusted<T>\`. The runtime then
  checks and repairs them, with no instructions needed.
- Refining until good enough is \`iterateOn\`: \`\`iterateOn(revise, draft).until(nl\`draft answers every question in
  message\`)\`\`. The step function returns the next state, and a natural-language stopping check is allowed.
- The same function under another policy is \`fn.in(context)\`; on a stronger model it is \`fn.on({ model: NAME })\`.
  Gate on \`decide\`'s confidence and send only the unsure cases there.
- A decision with both a code rule and a judgment uses \`pluggable({ crisp, nl }, setting)\`. In \`shadow\` mode both
  run, and their disagreements are recorded.
- Repeated inputs compile to code: \`natlang specialize\` turns a function's recorded calls into guarded crisp cases
  that answer without the model, and hands every other input back to it.
- The program improves from its runs: \`natlang tune\` trains the local model's weights on this program's examples (it
  needs a native student model; \`natlang doctor\` says whether one is set up), and \`natlang improve init --export plan\`
  writes an improvement case: give it train, validation and test cases, and \`natlang improve CASE.json --out DIR
  --model MODEL\` searches for better source against them.

## The model

On exit code 3, ask the user before running \`natlang setup --yes\` (a one-time 1.8 GB download), or use a hosted
model: \`--provider ID\` (\`natlang models\`) or \`NATLANG_SERVER=URL NATLANG_MODEL=ID\`. \`natlang doctor --json\` reports
the setup.

The natlang-authoring skill (\`natlang skills install\`) has the full language: composition, transactions over
folders, Neuralese values, recursion and decisions.
`;

/** The one-function program (`natlang init --template classifier`, and the dashboard's Playground instructions). */
function classifierProgram(name: string, options: { input?: string; profile?: string; custom?: { instruction: string; returns: string } }): Record<string, string> {
  const { custom } = options;
  const input = options.input ?? sampleInput;
  const parameter = custom ? 'input' : starter.parameter;
  // One function, its manifest and its tests: the package's main target is answer.nl itself, so there is no entry
  // module, tsconfig or input file to learn before the first edit. AGENTS.md (and CLAUDE.md, which imports it) tells a
  // coding agent the non-interactive loop. `answer/` (its helpers, once added) is part of the package; package.json makes
  // TypeScript helpers and a main.ts ES modules.
  return {
    'answer.nl': answerSource(custom),
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name, version: '0.1.0', description: 'My first natlang program.',
      ...(options.profile ? { model: { profile: options.profile } } : {}),
      include: ['answer', 'answer.nl', 'examples.json', 'package.json', 'README.md'],
      targets: { main: { entry: 'answer.nl', description: 'Run answer.nl on the sample input, or supply your own text.',
        arguments: [{ name: parameter, type: 'string', description: 'Text to process; leave blank for the sample.', default: input }] } },
      warmup: [{ name: 'sample input', function: 'answer.nl', inputs: { [parameter]: input } }] }, null, 2) + '\n',
    'examples.json': JSON.stringify(custom ? [] : [{ name: 'duplicate charge', function: 'answer.nl', inputs: { [parameter]: input }, expected: 'billing' }], null, 2) + '\n',
    'package.json': JSON.stringify({ name, private: true, type: 'module' }, null, 2) + '\n',
    '.gitignore': '.natlang/\nnode_modules/\n*.d.nl.ts\n',
    'README.md': programReadme(name, parameter),
    'AGENTS.md': agentGuide(parameter),
    'CLAUDE.md': '@AGENTS.md\n',
  };
}

const programReadme = (name: string, parameter: string) => `# ${name}

${fence}sh
natlang run                          # answer.nl on the sample input
natlang run . -- "My own text"       # or: natlang call answer.nl "My own text"
natlang dev                          # rerun after every edit; Ctrl+C stops
natlang test                         # check the saved examples
natlang dashboard --open             # results, traces and model settings in the browser
${fence}

**answer.nl** is a natural-language function: a typed signature (\`args\`, \`returns\`) and instructions that a model
carries out. Edit the instructions to change its behavior. **examples.json** saves inputs with their expected results, and
\`natlang test\` runs them on the real model. The sample input is the \`${parameter}\` argument's default in **natlang.json**.
Helpers that answer.nl may call go in the **answer/** folder; **package.json** makes TypeScript files ES modules.

\`natlang check\` checks the source without calling a model. Every call prints its call ID on stderr: \`natlang traces show
CALL_ID\` shows the steps the model took. The first run prepares the selected model: the local one is a one-time download
(1.8 GB), or choose a hosted model on the dashboard's Models page. AGENTS.md is the guide for coding agents.
`;

/** The coding-agent guide every new program carries (AGENTS.md; CLAUDE.md imports it). */
const agentGuide = (parameter: string) => `# Working on this natlang program

natlang is TypeScript with natural-language functions. \`answer.nl\` is one: YAML front matter declares its typed
signature (\`args\`, \`returns\`), and the body is instructions that a model executes. The signature is checked like
TypeScript; the instructions' behavior is checked by running examples on the model.

## Files

- \`answer.nl\`: the function. Edit the instructions to change behavior, \`args\`/\`returns\` to change the contract.
- \`examples.json\`: saved cases, \`[{ name, function: "answer.nl", inputs: { ${parameter}: ... }, expected }]\`, compared exactly.
- \`natlang.json\`: the package. \`targets.main\` runs \`answer.nl\`; its \`${parameter}\` argument's default is the sample.
  \`include\` lists what the package holds (\`answer/\` included).
- Generated (do not edit): \`*.d.nl.ts\` and \`.natlang/\`.

## The loop

1. \`natlang check --json\`: types, signatures and call sites, without a model. Prints
   \`{ ok, diagnostics: [{ file, line, column, code, message, hint? }], functions: [{ source, signature }] }\`; fix every
   diagnostic before calling.
2. \`natlang call answer.nl "some text" --json\`: one real call; arguments fill parameters in order. stdout is the
   result as JSON; stderr names the call: \`natlang: call CALL_ID (natlang traces show CALL_ID)\`.
3. \`natlang test --json\`: every example in examples.json, with expected and received values and each one's \`call_id\`.
4. \`natlang traces show CALL_ID\`: each step the model took, to see why a result is wrong. \`natlang traces fix CALL_ID
   --expected JSON\` saves that call's inputs with the corrected result into examples.json.

Exit codes: 0 ok; 1 the program is wrong (diagnostics, failed examples, a call that failed); 2 usage error (bad flags,
arguments or paths); 3 the model is unavailable (runtime not installed, endpoint unreachable, missing credentials).
Model calls take seconds on a hosted model and can take minutes on a local CPU, so check first and call deliberately.
When you change behavior, add or update an example and run \`natlang test\`.

## The model

On exit code 3, ask the user before running \`natlang setup --yes\` (a one-time download of 1.8 GB), or use a hosted
model: \`--provider ID\` (\`natlang models\` lists them) or \`NATLANG_SERVER=URL NATLANG_MODEL=ID\` for an
OpenAI-compatible endpoint. \`natlang doctor --json\` reports the setup.

## Writing natlang

- Exact work (arithmetic, parsing, sorting, I/O) belongs in TypeScript; judgment belongs in instructions.
- Put constraints in types: a string-literal union or a record in \`returns\` beats a sentence asking for a format.
- \`readout: decision\` (in answer.nl) makes the model reason, then score each value of a finite \`returns\` (a literal union,
  \`boolean\`, or finite record): calibrated, for a choice that gates what happens next (\`readout: instant\` skips reasoning
  on hot paths). Such a call gets no tools, so a function that must call helpers or compute first leaves it out.
- Helpers the function may call go in its companion folder, \`answer/\` (\`answer/lookup.ts\`, \`answer/summarize.nl\`).
- When the instructions are several steps of judgment, number them and name the data each step uses; the executing
  model may be small. Exact steps go in a TypeScript helper instead.
- Language reference and patterns: the natlang-authoring skill (\`natlang skills install\` adds it to Claude Code or Codex).
`;

/**
 * Add a natural-language function to a program from a template: `file` is a new `.nl` path inside it, such as
 * `classify.nl` or `answer/lookup.nl` (a helper in answer.nl's callable folder). Existing files are never overwritten.
 */
export function createFunction(root: string, file: string): string {
  if (!/^([A-Za-z_][A-Za-z0-9_]*\/)*[A-Za-z_][A-Za-z0-9_]*\.nl$/.test(file))
    throw new Error(`${file} is not a function file name: use letters, digits and underscores ending in .nl, such as classify.nl or answer/lookup.nl`);
  const path = join(resolve(root), ...file.split('/'));
  if (existsSync(path)) throw new Error(`${file} already exists. Choose a new name; existing files are never overwritten.`);
  mkdirSync(join(path, '..'), { recursive: true });
  writeFileSync(path, `---\ndescription: Say in one sentence what this function decides or produces.\nargs:\n  text: string\nreturns: string\n---\n` +
    'Say what to decide or produce from text. Keep exact steps in a TypeScript helper this function calls.\n', { flag: 'wx' });
  return path;
}

