/**
 * The proposing side of `natlang improve DIR` (WS8 N4): a natural-language improver that reads the program's own
 * recorded runs and proposes a change set, which `improveDirectory` then checks and adopts with its exact gate.
 *
 * Decisions are natural language, run on the configured model (LFM3 by default), as built-in programs whose calls are
 * recorded like any other (src/builtin):
 *   - `improveDiagnose.nl`: what the troubled calls of one function have in common, and which instruction is at fault;
 *   - `improveShouldRevise.nl` (`readout: decision`): whether that diagnosis is worth a revision;
 *   - `improveRevise.nl`: the function's new instructions.
 * Crisp mechanism: reading the evidence from the call store (run-evidence.ts), running the support split to find the
 * examples the program answers wrongly, keeping each file's front matter byte-identical (signature, `protect:`,
 * `ensures:` stay the author's), and assembling the change set.
 */
import { builtin } from '../runtime/builtin.js';
import { gatherRunEvidence, hasActionableEvidence, type FunctionEvidence, type RunEvidenceStore } from './run-evidence.js';
import { DIRECTORY_GUIDANCE_FILE, parseDirectoryGuidance, type DirectoryGuidance, type DirectoryImprovementCase, type DirectoryProposal, type DirectorySplitResult } from './program.js';

export type ExampleMiss = { inputs: unknown; expected: unknown; received?: unknown; error?: string };
export type ImproverEvidence = Omit<FunctionEvidence, 'function'> & { wrong: ExampleMiss[] };

export type ProposeFromRunsOptions = {
  /** The call store to read the program's runs from. Without one, only the support examples are evidence. */
  store?: RunEvidenceStore;
  /** Run the current program on support cases (the examples it answers wrongly are evidence). */
  evaluateSupport?: (cases: readonly DirectoryImprovementCase[]) => Promise<DirectorySplitResult>;
  /** Paths the improver may not change (protected sources, `types.ts`). */
  protectedPaths?: ReadonlySet<string>;
  /** At most this many functions are revised in one change set (default 3), the most troubled first. */
  maxFunctions?: number;
  /**
   * What kind of improvement to propose: `'instructions'` (default: revise `.nl` instructions, falling back to
   * `guidance.json` for protected functions when `allowGuidanceFallback` is set), `'guidance'` (propose persisted
   * memetic guidance in `guidance.json` without editing `.nl` files), or `'both'`.
   */
  mode?: 'instructions' | 'guidance' | 'both';
  /** Replacements for the built-in stages (tests, or another improver program). */
  stages?: Partial<ImproverStages>;
};

export type FunctionUnderReview = { path: string; declaration: string; instructions: string };
export type ImproverStages = {
  diagnose(target: FunctionUnderReview, evidence: ImproverEvidence): Promise<string>;
  shouldRevise(target: FunctionUnderReview, evidence: ImproverEvidence, diagnosis: string): Promise<boolean>;
  revise(target: FunctionUnderReview, evidence: ImproverEvidence, diagnosis: string): Promise<string>;
  reviseGuidance?(target: FunctionUnderReview, evidence: ImproverEvidence, diagnosis: string, existing?: string): Promise<string>;
};

/** What the improver decided about one function, for the report. */
export type FunctionDecision = { path: string; diagnosis: string; revise: boolean; changed: boolean; problem?: string; guidance?: boolean };
export type RunsProposal = DirectoryProposal & { decisions: FunctionDecision[]; evidence: Record<string, ImproverEvidence> };

const FRONT = /^(---\r?\n[\s\S]*?\r?\n---\r?\n?)([\s\S]*)$/;

const shippedStages = (): ImproverStages => ({
  diagnose: (target, evidence) => builtin('improveDiagnose')(target, evidence) as Promise<string>,
  shouldRevise: (target, evidence, diagnosis) => builtin('improveShouldRevise')(target, evidence, diagnosis) as Promise<boolean>,
  revise: (target, evidence, diagnosis) => builtin('improveRevise')(target, evidence, diagnosis) as Promise<string>,
});

/** How much a function's runs ask for attention: wrong answers and disagreements weigh most. */
const weight = (evidence: ImproverEvidence) => 3 * (evidence.wrong.length + evidence.feedback.length +
  evidence.escalations.filter(item => !item.agreed).length) + 2 * evidence.repairs.length + evidence.failures.length;

function existingGuidanceFrom(files: Readonly<Record<string, string>>): DirectoryGuidance {
  const raw = files[DIRECTORY_GUIDANCE_FILE];
  if (!raw) return {};
  try { return parseDirectoryGuidance(raw); } catch { return {}; }
}

/**
 * Propose a change set for the program in `files` from its recorded runs and its support examples. Returns null when
 * nothing in the evidence calls for a change. Must run inside a natlang task (the stages are natural-language calls).
 */
export async function proposeFromRuns(context: { directory: string; files: Record<string, string>; support: readonly DirectoryImprovementCase[] },
    options: ProposeFromRunsOptions = {}): Promise<RunsProposal | null> {
  const stages = { ...shippedStages(), ...options.stages };
  const mode = options.mode ?? 'instructions';
  const evidence: Record<string, ImproverEvidence> = {};
  const entry = (path: string) => evidence[path] ??= { calls: 0, failures: [], escalations: [], repairs: [], feedback: [], wrong: [] };
  if (options.store) for (const [path, found] of Object.entries(gatherRunEvidence(options.store, context.directory))) {
    const { function: _path, ...rest } = found;
    Object.assign(entry(path), rest);
  }
  if (options.evaluateSupport && context.support.length) {
    const result = await options.evaluateSupport(context.support);
    const byId = new Map(context.support.map(item => [item.id, item]));
    for (const row of result.results) {
      if (row.ok) continue;
      const item = byId.get(row.id);
      if (!item) continue;
      const target = entry(item.function);
      if (target.wrong.length < 6) target.wrong.push({ inputs: item.inputs, expected: item.expected,
        ...(row.error !== undefined ? { error: row.error } : { received: row.received }) });
    }
  }
  const candidates = Object.entries(evidence)
    .filter(([path, found]) => Object.hasOwn(context.files, path) && FRONT.test(context.files[path]!) &&
      (mode !== 'instructions' || !options.protectedPaths?.has(path)) &&
      (found.wrong.length > 0 || hasActionableEvidence({ function: path, ...found })))
    .sort(([a, x], [b, y]) => weight(y) - weight(x) || a.localeCompare(b))
    .slice(0, options.maxFunctions ?? 3);
  const files: Record<string, string> = {};
  const decisions: FunctionDecision[] = [];
  const nextGuidance: DirectoryGuidance = existingGuidanceFrom(context.files);
  let guidanceChanged = false;
  for (const [path, found] of candidates) {
    const [, front, body] = FRONT.exec(context.files[path]!)!;
    const target: FunctionUnderReview = { path, declaration: front!.replace(/^---\r?\n|\r?\n---\r?\n?$/g, ''), instructions: body!.trim() };
    const diagnosis = String(await stages.diagnose(target, found)).trim();
    const revise = await stages.shouldRevise(target, found, diagnosis) === true;
    if (!revise) { decisions.push({ path, diagnosis, revise, changed: false }); continue; }
    const canEditSource = !options.protectedPaths?.has(path) && (mode === 'instructions' || mode === 'both');
    const shouldWriteGuidance = mode === 'guidance' || mode === 'both' || !canEditSource;
    let editedSource = false;
    let editedGuidance = false;
    let problem: string | undefined;
    if (canEditSource) {
      const revised = String(await stages.revise(target, found, diagnosis)).trim();
      // Exact checks on what the stage returned: some text, a change, and no front matter of its own (the declaration
      // stays the author's, so a revision cannot change the signature or what is protected).
      problem = !revised ? 'the revision is empty' : revised === target.instructions ? 'the revision repeats the instructions' :
        /^---\r?\n/.test(revised) ? 'the revision carries its own front matter' : undefined;
      if (!problem) {
        files[path] = `${front}${revised}\n`;
        editedSource = true;
      }
    }
    if (shouldWriteGuidance) {
      const prev = nextGuidance.functions?.[path];
      const proposedNote = stages.reviseGuidance
        ? String(await stages.reviseGuidance(target, found, diagnosis, prev)).trim()
        : diagnosis;
      if (!proposedNote) {
        if (!editedSource) problem ??= 'the guidance note is empty';
      } else if (proposedNote === prev) {
        if (!editedSource) problem ??= 'the guidance repeats existing guidance';
      } else {
        nextGuidance.functions = { ...(nextGuidance.functions ?? {}), [path]: proposedNote };
        guidanceChanged = true;
        editedGuidance = true;
      }
    }
    const changed = editedSource || editedGuidance;
    decisions.push({ path, diagnosis, revise, changed, ...(editedGuidance ? { guidance: true } : {}), ...(!changed && problem ? { problem } : {}) });
  }
  if (guidanceChanged) files[DIRECTORY_GUIDANCE_FILE] = `${JSON.stringify(nextGuidance, null, 2)}\n`;
  if (!Object.keys(files).length) return candidates.length ? { files: {}, decisions, evidence, summary: 'no revision proposed' } : null;
  return { files, decisions, evidence,
    summary: decisions.filter(item => item.changed).map(item => `${item.path}${item.guidance ? ' (guidance)' : ''}: ${item.diagnosis}`).join('\n') };
}
