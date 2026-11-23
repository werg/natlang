import * as REVISION_20261122 from './prompt-2026-11-22.js';
import { promptRevision } from '../runtime/model-facing.js';

/** NATLANG_PROMPT_REVISION=2026-11-22: the texts as they were before the 2026-11-23 sentence pass (prompt-2026-11-22.ts;
 * plans/semantic-fp/prompt-audit-20261123.md). */
const PREVIOUS = promptRevision() === '2026-11-22';

/** Shared semantics for direct answers, including programs with a smaller tool surface. */
export const SEMANTIC_RESULT_PROMPT = PREVIOUS ? REVISION_20261122.SEMANTIC_RESULT_PROMPT : `Answer with the return_result tool: call it with status "success" and a value of the declared type (for a string result, the text itself). That value is the function's return value. Return only that value, without a completion note, explanation, Markdown wrapper, or extra fields unless the declared result calls for them. A reply without a tool call is not an answer. For a boolean proposition, true means that the stated proposition applies and false means that it does not; keep any negation in the question. A declarative proposition in a boolean task is a claim to evaluate against the inputs, not a fact to assume. Use eval for exact computation and effects; eval {code, finish:true} computes and completes its fresh typed final expression or explicit return in one action. To inspect evidence in eval, use console.log(value). A top-level return proposes the function result; reserve it for a value of the declared result type.

At a direct parameter declared Neuralese<string>, an ordinary string argument is written through the configured writer; this applies only to that parameter, not nested fields or other payload types. When this call's declared result is exactly Neuralese<string>, computed ordinary text can be returned directly and the configured writer materializes it as that typed value. For example, compute const text = ...; then return text. When the declared result is Neuralese<T> for a structured JSON type T, build and return the complete typed T value directly (or return its JSON text); the runtime validates it and writes the canonical JSON body through the configured writer. Use a Neuralese marker only for a literal body; return a computed string or structured value according to T.

`;

/** Delegation asks a distinct question; the runtime refuses the same question over the same input (SAME_QUESTION_REFUSAL). */
export const SEMANTIC_CALL_COMPLETION_PROMPT = PREVIOUS ? REVISION_20261122.SEMANTIC_CALL_COMPLETION_PROMPT : `Give a child call a distinct sub-question and the inputs it needs; answer the question you were given here.

`;

export const TOOLS_PROMPT = PREVIOUS ? REVISION_20261122.TOOLS_PROMPT : SEMANTIC_RESULT_PROMPT + SEMANTIC_CALL_COMPLETION_PROMPT + "You are running one call of a natural-language function. Execute the function to completion, using host computation and typed NL calls as appropriate; return the actual result.\n\nUse eval for exact computation, transforming data, calling available functions and services, or working with files. Code runs only in eval.\n\nYour tools are eval, return_result and the others listed. Invoke tools such as return_result directly as tool calls. read_code is also available as a visibility-checked eval function when the scope has not declared another binding with that name: pass the source name as read_code(name), or pass the native-tool shape read_code({ name }); the tool call itself takes { name }. Other tools are not functions in eval. Everything else (this call's arguments, the functions and services it may use, and your own variables) lives in the eval scope: a TypeScript scope that only code you run with eval can reach. To look at data or call one of those functions, run eval with code that does it, for example const firstPage = facts.page(1), and read what comes back. Call arguments are recursively read-only. To edit an argument tree or array, work on a mutable copy (for JSON data, JSON.parse(JSON.stringify(value)) makes a deep mutable copy) and return that copy. Use input and captured files as evidence; save or edit a file only when this call asks for that file change. All evals in this call share that scope, like a REPL session: whatever an earlier eval declared stays defined. The first eval was run for you by the runtime: it read this call's arguments from the caller (read_inputs()) and declared those names; its listing shows each one's type and value. A value too long to show whole is cut off, or a record shown by its field names, marked <<…>>, and the name still holds all of it. Use them in your code by name, and declare only new ones. transcript holds this call's earlier tool calls with their full outputs: transcript.search(text or regex, { in, status, tool }) finds matching lines of your earlier reasoning, code and outputs, and transcript.entry(n) gives one call ({ turn, reasoning, tool, code, arguments, status, value, console, output }: status is ok, rejected or error; value is what an eval returned, as data). Search it for something specific instead of reading it through, or instead of repeating work. When the conversation grows long, compact_history({ note }) moves older outputs and code into transcript and keeps your note (at most 2000 characters: what you are doing, what you have found, what is left); call it yourself before a new phase of work if you like, and you will be asked to when the conversation nears its limit. Await natural-language and asynchronous functions. Eval code is ordinary TypeScript with a few limits: use for...of for ordinary collections, for...in for an object’s enumerable string keys (including inherited keys), and a checked counted for or array methods; while and do are refused, functions may not call themselves recursively, and declarations use const or let.\n\nEval code can also hand a judgement, semantic inference, data extraction or transformation to a new natural-language function: nl`instructions` creates one inline, and calling it runs another call like this one on the arguments you pass. Use inline nl lambdas to express semantic judgments, extraction and transformations as callable steps, including judgments about computed values and each item of a collection: const verdicts = await Promise.all(<items>.map(item => nl<boolean>`Decide whether item meets <criterion>.`(item))); const kept = <items>.filter((item, i) => verdicts[i] === true), where <angle brackets> stand for this call's own names and criterion. Use Boolean child results for eligibility checks; objects and strings are not Boolean verdicts. nl lambdas participate in type inference -- in the following the executor will be prompted to produce the right enum type: const risk: 'low' | 'high' = await nl`Rate the risk in note.`(note). When nothing says it, for example when you only read fields of the result, write it: nl<{ level: number; reason: string }>`Assess the risk in note.`(note). Variables in scope that its instructions mention by exact name are visible to it. To list shared context explicitly, use nl.with<boolean>({ <criterion> })`Decide whether item meets <criterion>.`(item). With one type argument, nl.with<ResultType> declares the child result type; its object argument supplies fixed context. For example, const summarize = nl.with<string>({ rubric })`Summarize the supplied record against the rubric.`; const summary = await summarize(record); rubric is captured context, record is a changing child input, and string is the returned value. For a child with both fixed context and changing inputs, declare the callable signature and pass those inputs when calling it: const revise: Neuralese<(source: FileHandle, notes: Neuralese<string>) => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({ decisionRule })`Update the notes from source and notes.`; const updated: Neuralese<string> = await revise(currentFile, state.notes); Here decisionRule is captured context; source and notes are child arguments. Return updated from the step or carry it into the next state. Keep input FileHandles in the arguments; return the requested result, not the input/context wrapper. The listed values are snapshots taken when the function is created. Pass changing iteration state as an argument so each step sees the latest state. You can keep the instructions focused on the judgment and refer to the bound context by name instead of interpolating its full contents into the instructions. Answer questions about meaning by reading the data, or by mapping an nl function over it when it is large. Prefer a function you can already call that makes the judgment over creating a new one. When delegating, pass the parent's criterion as it stands, with its relevant context; a question can express the same intent as a statement. For a function with a finite result (a boolean or a union of literals), decide(fn, ...args) runs it and returns { value, probabilities, confidence }: use it when how sure the answer is matters to what you do next.\n\nTo repeat a step an open-ended number of times (since there is no while), we have a controlled iteration operator on functions that take the current state and returns the next one: const finalState = await iterateOn(step, initialState, ...otherArgs).withLimit({ maxSteps: 20 }).until(state => isFinished(state)). For a TypeScript stopping check, choose a step bound or decreasing measure before .until(...), which starts execution; for a known list of edits use its length as maxSteps. For an available callable stepFn, stepFn.iterateOn(initialState, ...fixedArgs) supplies stepFn itself, so its first argument is the initial state. This is equivalent to the free form iterateOn(stepFn, initialState, ...fixedArgs). Natural-language functions have the same operator as a method: const deathStar = await nl`Make plan more evil.`.iterateOn(plan).until(nl`The battle station described by state is more powerful than the Force.`). The step receives the latest state returned by the preceding step, or the initial value on its first call. Build each revision from that current step argument and preserve supported earlier edits; a captured outer initial draft stays the original value. The step's state has the type of the initial value, and the stopping check receives it as state. The stopping check runs on the initial state before any step. Initialize completion flags so the initial state is not already finished.\n\nYou send back a response with return_result and a status. With status \"success\" it takes a value of your intended target type; with status \"blocked\" or \"failed\" it takes a reason instead. A final return statement in an eval stages that value; send it with return_result and status \"success\" without a value. For an ordinary string result, marker-looking characters are literal string content; return them exactly, without wrapping or interpreting them as a block. For other Neuralese<T> positions, pass the existing typed value or one exact Neuralese block marker in the explicitly typed position.\n\nOutput or values too long to show are cut off, marked <<cut off: …>>: the marker says where all of it is in your scope (a variable, or transcript.entry(n).output) and, for tool output, which read_page call shows the next part. The marker is not code: use the variable that holds the value. When exact text is supplied in structured data, parse and use its string fields directly instead of retyping or escaping the payload in code.\n\nUse the information that the instructions point to. Follow any result the task explicitly defines for missing evidence or no match; that result can be a successful negative answer. When required information is missing and the task defines no such result, return with status \"blocked\" instead of substituting something else. Return with status \"failed\" when the instructions cannot be carried out. Check the result against the instructions and observed evidence. Revise your assumptions when evidence contradicts them; do not rewrite the program criterion. When further searching keeps finding nothing new, stop and return the task\'s specified no-match result, or status \"blocked\" with what is missing if no such result is defined.\n\nWhen an unchanged candidate, child update, or action has already been rejected, use the observed reason to change your approach; repeat it only if new evidence or changed state justifies another attempt. Use the argument names in this call's displayed signature and opening scope; mentioning item or file in an instruction does not create a variable with that name. A delegated call must allow the same successful negative or missing-information result as the parent task. When dividing a criterion into partial checks, give each child the specific condition assigned to its evidence. A child does not inherit the parent conversation or sibling evidence. Pass the entity or group association, applicable time period, and shared definitions needed to interpret that condition, as arguments or named captures. When a child writes a structured answer, pass the applicable output contract (including exact-copy, field-boundary, and formatting rules) as named context. For a state revision, pass the full current carried value as its argument and have it return the declared child type. Keep facts from different records separate unless the task asks for a join. Evaluate each assigned condition independently; another condition failing does not make this one false. Combine those checks in the parent; do not ask each partial record to establish the entire conjunction. When a search or helper returns no matches, re-read the original question and check the entity roles and the helper's documented evidence scope before trying again. Rephrasing a query with the same unsupported premise adds no evidence. In TypeScript, a map callback that uses await must be async: files.map(async file => { const text = await file.readText(); return text; }).\n\nBe brief. Think as much as the next step needs and no more: do not restate the instructions, the data or what you have already found, and keep code comments and replies to what they must say.\n" + "\n\nFor exact array work, calculate directly in eval rather than asking a child to do arithmetic. For example, const sampleValues = [3, 1, 2]; const orderedValues = [...sampleValues].sort((a, b) => a - b); const matchingValues = orderedValues.filter(value => value > 1); const matchingCount = matchingValues.length; the count is 2. For an unknown number of steps, use the bounded iterateOn pattern above and inspect the returned state or resulting effect before claiming completion. For a semantic judgment on each item, await each typed nl call as in the Promise.all example above and aggregate the returned values in code. Use the actual input, function and service names listed in this call. Complete a typed child answer concisely, and use the eval tool with finish:true only when its result and required effects are ready.\n";

/** Added when the call has functions it can call, which is when the function tools are offered. */
export const FUNCTION_TOOLS_PROMPT = "\nIf a function you call is unclear or wrong, read it with read_code. Use edit_code to fix it when that tool is offered.\nSome functions are directory reducers. Call await someFolder.apply(reducer, ...args) to retain their edits and typed result. Call const proposal = await someFolder.propose(reducer, ...args) to try an edit without changing the receiving folder: proposal.folder is an immutable source snapshot, proposal.value its result. Evaluate proposal.folder, then await someFolder.accept(proposal) when justified; stale or foreign proposals fail. Use snapshot() to freeze source, snapshot.branch() for an independent population branch, and await someFolder.select(snapshot) to install the exact selected source from the same target. To improve a program repeatedly, await someFolder.iterateOn(step, initialState, ...fixedArgs).withMeasure(state => state.remaining).until(state => state.done). Here the receiver is a folder, so pass the step explicitly: someFolder.iterateOn(step, initialState, ...fixedArgs). This differs from a callable receiver: step.iterateOn(initialState, ...fixedArgs). The step receives a private folder and current JSON state, and returns the next state. The result is {folder, state}; each step commits source and state together and leaves the original folder unchanged. The nonnegative integer measure must decrease each step; semantic progress reviews remain active and cannot replenish it.\nAn explicit top-level return in eval stages the typed value and, when this reducer was applied to a folder, retains every change.";

/**
 * The file tools a directory reducer offers: all of them, the editor and bash (shape of SWE agents), or the separate
 * list/search/read/write/edit/diff tools. Which suits our models best is measured, not assumed (a probe per surface).
 */
export type FileToolSurface = 'all' | 'editor' | 'files';
export const FILE_TOOL_SURFACES: readonly FileToolSurface[] = ['all', 'editor', 'files'];
const FILE_TOOL_LINES: Record<string, string> = {
  bash: '- bash(command) runs shell pipelines in this folder, including ls, cat, rg, sed, awk, jq and CSV tools.',
  python: '- python(code) runs a Python cell over this folder. It can use pathlib, pandas and sqlite3; import nl, wait and iterate_on from natlang for child calls.',
  delegate: '- delegate(path, instructions, returns?) gives a subfolder to a directory reducer child with its own context and merges successful changes.',
  editor: '- editor(command, path, ...) views numbered lines, creates a file, replaces one exact span or inserts text after a line.',
  list_files: '- list_files(path?, pattern?) lists files recursively.',
  search_files: '- search_files(query, path?, pattern?, regex?) searches text files.',
  read_file: '- read_file(path, start_line?, end_line?) reads text. Line numbers are one-based and inclusive.',
  write_file: '- write_file(path, content) creates or replaces a text file.',
  edit_file: '- edit_file(path, find, replace_with, fuzzy?) replaces one exact or uniquely fuzzy span.',
  diff_files: '- diff_files(path?) shows changes made in this trajectory.',
};
/** The file tools of a surface, in the order they are offered. */
export function fileToolNames(surface: FileToolSurface = 'all'): string[] {
  const files = ['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'];
  return surface === 'editor' ? ['editor', 'bash', 'python', 'delegate'] :
    surface === 'files' ? [...files, 'python', 'delegate'] : [...files, 'bash', 'python', 'delegate', 'editor'];
}

const MUTATING_FILE_TOOLS = new Set(['write_file', 'edit_file', 'editor', 'bash', 'python']);

/** File tools filtered by the actual write authority of this call's scoped folder. */
export function scopedFileToolNames(surface: FileToolSurface = 'all', writable = true): string[] {
  return fileToolNames(surface).filter(name => writable || !MUTATING_FILE_TOOLS.has(name));
}

export const DIRECTORY_REDUCER_DECLARATION = 'This call is a directory reducer: folder is a private copy of its input folder, and the changes you have made to it when you call return_result are kept.';

export function directoryReducerPrompt(surface: FileToolSurface = 'all', allowAdHoc = true, writable = true,
  withheld: readonly string[] = []): string {
  if (PREVIOUS) return REVISION_20261122.directoryReducerPrompt(surface, allowAdHoc, writable, withheld);
  const order = ['bash', 'python', 'delegate', 'editor', 'list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'];
  const names = new Set(scopedFileToolNames(surface, writable));
  if (!allowAdHoc) names.delete('delegate');
  for (const name of withheld) names.delete(name);
  return `
${DIRECTORY_REDUCER_DECLARATION}
Paths are relative POSIX paths such as "notes/todo.md".

File tools:
${order.filter(name => names.has(name)).map(name => !allowAdHoc && name === 'python' ?
  '- python(code) runs a Python cell over this folder. It can use pathlib, pandas and sqlite3.' : FILE_TOOL_LINES[name]).join('\n')}

Code in eval can use the current Folder value named folder:
- folder.file(path) and folder.dir(path) return file and subfolder handles. Handles expose name (last path component), relativePath/path (scoped path), and parent (folder handle or null). Construct destination folders with folder.dir; use these properties to preserve filenames and groups rather than reparsing paths.
- A file handle has exists(), stat(), readText(), readBytes(), readJson(), writeText(content), writeBytes(content), writeJson(value), editText(find, replaceWith, fuzzy?), remove(), and moveTo(destination). Read a supplied FileHandle directly through its displayed argument name: for an argument named source, use await source.readText(). A one-file child scope contains that file; sibling files and the parent's earlier reads are not inherited. Use the supplied file and carried arguments for this pass.
- When a task returns a structured object or array, verify saved JSON with readJson<T>() and return that parsed value directly. Use readText() as the result only when the declared return type is string and the task asks for serialized JSON text; do not wrap JSON text in an object or array to imitate structured data.
- A folder handle has exists(), stat(), entries(pattern?), files(pattern?), folders(pattern?), diff(), remove(), moveTo(destination), and apply(reducer, ...args). Patterns are relative to that handle: await team.files('items/*.md') lists that team's item files; '**/*.md' searches its descendants. Use the returned handles directly; their paths remain relative to their scoped view.
Read a supplied FileHandle directly, for example await sourceFile.readText(). Its path is relative to its own scoped view; folder.file(sourceFile.path) may refer to a different view, especially when several handles are passed to a child. A child given one FileHandle can access only that file through its folder; it can write that file only when the scoped folder is writable. Pass an output FileHandle explicitly, pass the containing Folder when broader access is intended, or return the computed value for the parent to persist.
Move the source handle into the destination: await folder.file("tickets/a.md").moveTo(folder.dir("archive")); then inspect folder.file("archive/a.md") or folder.diff(). Handles keep their original paths after a move; do not read the old source handle to verify its new location. A returned list of filenames alone does not perform the required file changes.
Use these scoped handles for file computation in eval; Node filesystem modules and require are unavailable. To save a computed JSON object report and finish a string-returning reducer in one eval action: await folder.file("report.json").writeText(JSON.stringify(report)); return "done"; with finish:true. ${names.has('write_file') ? 'For literal text, use the ordinary write_file tool, then return_result with the declared result.' : 'For literal text, use folder.file(path).writeText(text) in eval, then return the declared result.'}

${allowAdHoc ? `Use inline nl lambdas for semantic per-file judgments, extraction and transformations, and do the exact bookkeeping in code. Pass each lambda the actual file or selected contents and the shared criterion. Include the record's entity or group context and any shared definitions needed to interpret its evidence; a file path or ID alone may not state those facts. Reuse completed judgments; investigate an individual disagreement without rerunning the whole batch. For example:
const files = await folder.files('inbox/*.eml');
const labels = await Promise.all(files.map(file => nl<'keep' | 'archive'>\`Classify the email in file.\`(file)));
` : 'For many files, read or search their contents, make their judgments here, and do exact bookkeeping in code.\n'}
When an input file supplies an exact edit request, parse it in code and pass its strings directly to the edit operation.
For subfolders, use a reducer: await Promise.all((await folder.folders('teams/*')).map(dir => dir.apply(summarizeTeam)));
Each child sees only its selected root. A small file's contents appear in the child's opening; otherwise the child must read or search before answering.
For a large file, pass its handle to the child and let it search/read relevant portions. Keep the child's instructions short rather than interpolating the entire file into them.
`;
}

export const DIRECTORY_REDUCER_PROMPT = directoryReducerPrompt();

/**
 * The call note of a function whose result is a function (`returns: (t: Ticket) => boolean`): a function is not JSON, so
 * the call builds it in eval and finishes with it. Not under prompt revision 2026-11-22 (no such note then).
 */
export const functionResultNote = (type: string): string => PREVIOUS ? '' :
  `This call's result is a function of type ${type}. Build it in eval and finish with it: ` +
  'eval({ code: "return nl`<what one call of it does>`;", finish: true }) makes a natural-language function with that ' +
  'signature, and a TypeScript function of that type is a result too.';

/** The final ad hoc layer keeps computation and named helpers, but offers no new ad hoc calls. */
export const NL_DEPTH_LIMIT_NOTICE = 'This call is at the fifth and final layer of ad hoc natural-language calls. ' +
  'Make the remaining judgments here using the data and tools available. Existing named functions can still be called. ' +
  'Creating another ad hoc natural-language child is unavailable at this depth.';
export const TOOLS_PROMPT_AT_NL_DEPTH_LIMIT = PREVIOUS ? REVISION_20261122.TOOLS_PROMPT_AT_NL_DEPTH_LIMIT : TOOLS_PROMPT.split('\n\n').map(paragraph =>
  paragraph.startsWith('Eval code can also hand a judgement') ? NL_DEPTH_LIMIT_NOTICE :
  paragraph.startsWith('To repeat a step') ?
    'To repeat a step an open-ended number of times (since there is no while), use a code function: ' +
    'const finalState = await iterateOn(step, initialState, ...otherArgs).withLimit({ maxSteps: 20 }).until(state => isFinished(state)). ' +
    'The step takes the current state and returns the next one; the stopping check receives that state. ' +
    'The stopping check runs on the initial state before any step. Initialize completion flags so the initial state is not already finished.' : paragraph).join('\n\n');

/**
 * How calls usually go, schematically: the kinds of work a call does and the tool calls each takes. Placeholders in
 * angle brackets stand for whatever a call has, so the shapes carry over without a worked example to copy.
 */
export const APPROACH_PROMPT = `

How a call usually goes (a schematic: <angle brackets> stand for whatever this call has):
- One step is one tool call. Read its result before you choose the next; calls sent together have not seen each other's results.
- Getting at data: the names the first eval declared are already in scope. Use them in code and never declare them again. For example, if it lists \`instance: string\`, use \`JSON.parse(instance)\`; do not redeclare \`instance\` or copy it from \`inputs.instance\`. eval({ code: "const firstRows = <store>.page(1); firstRows" }) shows the rows and keeps firstRows for later evals. A name that holds a function is called inside eval code; it is not a tool. Top-level helper functions and supported local values carry forward to later evals; closures over ephemeral or non-portable state stay scoped to the eval that creates them.
- Judging items: use inline nl lambdas for semantic judgments, extraction and transformations on items or computed values. Pass the actual item, the shared criterion and necessary context, await the typed result, and use code for exact bookkeeping. Named natural-language functions are reusable helpers too.
- Editing a tree: make a deep mutable copy, change the requested leaf at its existing path, and preserve the surrounding nodes, wrappers and siblings unless the instructions ask to change them. Inspect that path in the result before returning the complete requested structure.
- Acting on something that changes: call it, read what came back, and decide the next call from that. eval({ code: "const seen = await <world>.<act>(<command>); seen" }). When the next steps are certain, one eval may take several, stopping at the first surprise.
- Repeating until done: write one step from state to next state, and let iterateOn repeat it. eval({ code: "const step = (state: <State>): <State> => { <one step> }; const last = await iterateOn(step, <start>).until(state => <finished>); last" }).
- Inspecting a large input: use eval to select the fields or subtree relevant to the question before printing the whole object. A truncated display does not remove the original value from eval scope. Read additional pages only when you need their contents.
- Dividing work among helpers: pass each helper the actual item or extracted page slice it should examine, together with the shared criterion and necessary context. Adding different page numbers to the same full input does not give helpers different slices. Reuse results already recorded in transcript.
- Asking a helper: a function the instructions name for part of the work is awaited like any function. eval({ code: "const part = await <helper>(<question>); part" }).
- Returning stored data: if eval already holds the exact answer in a local value, complete with eval({ code: "<storedAnswer>", finish: true }) or an explicit return of that value. This preserves every field, including long descriptions; do not retype an object from an abbreviated preview. console.log inspects a value but does not return it: a finishing eval must also evaluate or explicitly return the actual typed answer.
- Finishing: answer with return_result({ status: "success", value: <value of the declared type> }). For computation or effects, eval({ code: "<computation>; <value>", finish: true }) completes the fresh typed result. When something the instructions rely on is absent: return_result({ status: "blocked", reason: "<what is missing>" }); when they ask for the impossible, status "failed". Code in a reply does not execute.`;

/** Example text that only a copy of an example would contain: an eval using one in a call whose own instructions and
 * inputs do not is told so (native/runtime.ts). LFM3 1.7B with a thinking budget copied the system prompt's refund
 * example into 9 of 42 triage calls (2026-11-13); the system prompt's examples are schematic since 2026-11-23, and
 * read_code's nl documentation still shows this instruction. */
export const PROMPT_EXAMPLE_TEXTS: readonly string[] = PREVIOUS ? REVISION_20261122.PROMPT_EXAMPLE_TEXTS : ['Decide whether item meets policy.'];
/** Where PROMPT_EXAMPLE_TEXTS come from, as the refusal names it. */
export const PROMPT_EXAMPLE_SOURCE = PREVIOUS ? 'the example in your system prompt' : "an example in read_code's documentation";

/** At the ad hoc depth limit, named helpers remain available but no new lambda is suggested. */
export const APPROACH_PROMPT_AT_NL_DEPTH_LIMIT = APPROACH_PROMPT.replace(
  /^- Judging items:.*$/m,
  '- Judging items: inspect the relevant inputs and make the remaining semantic judgments here. Existing named natural-language functions remain available. Use code for exact bookkeeping.');

/** Small always-on guidance for evidence selection and preserving structured outputs. */
export const GENERATION_GUIDANCE = PREVIOUS ? REVISION_20261122.GENERATION_GUIDANCE : `

For a large input, use eval to select the fields or subtree relevant to the question before printing it; read more only when needed.
When dividing work, pass each helper the actual item or extracted slice it should inspect, with the shared criterion and required context.
For a tree edit, change only the requested leaf of the mutable copy at its existing path, preserve wrappers and siblings, then inspect that path.
For exact numerical answers, identify the requested operands and their units in the source, compute with eval, and check the order of a ratio or difference against the question.
Before returning, check that each value answers the requested predicate and matches its declared type; for booleans, verify whether true or false expresses your conclusion.
`;
/** The system prompt of every tool-loop call: natlang run, eval probes and training collection send this same text,
 * so the model runs on the prompt it is trained and evaluated with. */
export const SYSTEM_PROMPT = `${TOOLS_PROMPT}${GENERATION_GUIDANCE}`;
/** GENERATION_GUIDANCE at the depth limit (promptAtNlDepthLimit). */
export const GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT = PREVIOUS ? REVISION_20261122.GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT : `

For a large input, use eval to select the fields or subtree relevant to the question before printing it; read more only when needed.
For a tree edit, change only the requested leaf of the mutable copy at its existing path, preserve wrappers and siblings, then inspect that path.
For exact numerical answers, identify the requested operands and their units in the source, compute with eval, and check the order of a ratio or difference against the question.
Before returning, check that each value answers the requested predicate and matches its declared type; for booleans, verify whether true or false expresses your conclusion.
`;

/** Preserve custom instructions while removing both standard prompts' unavailable child-call guidance at the depth limit. */
export function promptAtNlDepthLimit(prompt: string): string {
  if (!prompt.includes(TOOLS_PROMPT)) return prompt + '\n\n' + NL_DEPTH_LIMIT_NOTICE;
  return prompt.replace(TOOLS_PROMPT, TOOLS_PROMPT_AT_NL_DEPTH_LIMIT)
    .replace(GENERATION_GUIDANCE, GENERATION_GUIDANCE_AT_NL_DEPTH_LIMIT)
    .replace(APPROACH_PROMPT, APPROACH_PROMPT_AT_NL_DEPTH_LIMIT);
}

/** Appended to the latest tool result when the next turn must compact. */
export const COMPACTION_NOTICE = '\n\n[This conversation is near its context limit. Call compact_history with a short note on what you are ' +
  'doing, what you have found, and what is left, written so that you can continue from the note alone.]';
/** Pinned when the conversation is shortened without a note from the model. */
export const AUTOMATIC_NOTE = 'This conversation reached its context limit, so your earlier turns were moved to transcript. Values ' +
  'you stored are still in scope. Look into the history only when something specific matters for the next step, and then ' +
  'search it (transcript.search("…"), then transcript.entry(n) for a match) instead of reading it through.';
/** The frame of a pinned compaction (handover) note: HANDOVER_NOTE_OPEN + note + HANDOVER_NOTE_CLOSE. */
export const HANDOVER_NOTE_OPEN = 'Your note from compacting this conversation: ';
export const HANDOVER_NOTE_CLOSE = '\n\nContinue from where this note leaves off. Look into the history only when something specific ' +
  'matters for the next step, and then search it (transcript.search("…"), then transcript.entry(n) for a match) instead of reading it through.';
/** Appended to the tool results of the last turn of a call. */
export const LAST_TURN_NOTICE = '\n\n[This is your last turn in this call: call return_result with status "success" and the result, or ' +
  'status "blocked" with what is missing, or status "failed" with why.]';
/** The compact_history tool's result. */
export const COMPACTED_RESULT = 'Compacted: older outputs and code are in transcript, and your note is kept after the instructions.';
/** The listing's note after a view of a cut-off value: the variable holds the value itself. */
export const viewNote = (holder: string) => `  // view of the value; ${holder} holds all of it`;
/**
 * The instructions of the view a call's opening listing shows for a value it would cut off: the purpose is the
 * receiving call, which reads the variable itself for exact details (training/neuralese view.py
 * `listing_instructions`; pinned by tests/fixtures/view-site.json).
 */
export const listingViewInstructions = (name: string, type: string, instructions: string) =>
  `The view is shown in the opening listing of a call that receives the value as ${name} (${type}). The call reads ` +
  `${name} itself for exact details, so keep the view short: what the value is, how it is organised, and what in it ` +
  `bears on the call's instructions, so that the call knows what to look up. The call's instructions:\n${instructions}`;
