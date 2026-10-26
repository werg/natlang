/**
 * On-policy correction rounds for `natlang tune` (plans/HIERARCHICAL_LORA_TUNING.md §4.4).
 *
 * After a training round the student (the parent release, strictly bound) runs the TRAINING roots that carry a root
 * oracle. For a root it gets wrong, the authorized teacher is handed the student's own run at a turn boundary: the
 * student's earlier turns are replayed call by call (the handoff mechanism of teacher/replay.ts: `callMatcher`,
 * `scriptedDriver`), so the program re-executes in the same world, and the teacher answers the boundary turn and every
 * turn after it. Boundaries are tried from the latest backwards, so the longest student prefix the teacher can still
 * repair is kept.
 *
 * Verification is exact before admission: every replayed turn and the boundary turn must observe what the student
 * observed (`observed`, as handoffs check failsInPlace), and the repaired run must meet the root oracle. Only the
 * teacher's turns become training rows, tagged with their correction provenance; `tune/rows.py` renders their prompts,
 * which hold the student's mistaken prefix, at zero weight, so the prefix is context and never a target. The whole
 * repaired trajectory, with each turn's role, is kept in the round's records. Development, confirmation and protected
 * roots are never run here; rounds compare releases on development roots only.
 */
import { createHash } from 'node:crypto';
import { appendFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import type { WeightBinding } from '../adaptation/weights/binding.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { ModelDriver, Services } from '../runtime/runtime.js';
import { openingText, type Message } from '../teacher/opening.js';
import { observed, scriptedDriver } from '../teacher/replay.js';
import { collect, type CollectedTurn, type SplitExample } from './collect.js';
import { replayServices } from './world.js';

export const REPAIR_SCHEMA = 'natlang.tune-repair/v1';

/** Provenance of one correction row (plan §4.4 "On-policy rows also record ..."). */
export type CorrectionProvenance = {
  readonly round: number; readonly kind: 'repair' | 'continue';
  /** The boundary in the student's run: its flat turn index and the call/turn it was. */
  readonly boundary: { readonly index: number; readonly call: string | null; readonly turn: number };
  readonly student: { readonly checkpointSha256: string; readonly release: string; readonly actor: string;
    /** Requested/effective stack identity of the boundary call as the student ran it. */
    readonly stack: CollectedTurn['stack'] | null };
  readonly teacher: string;
  /** The original call ID of this teacher turn in the repaired run (rows use a repair-scoped call ID). */
  readonly call: string | null;
  readonly verification: { readonly studentRootOracle: false; readonly rootOracle: true; readonly prefix: 'observed-identical';
    readonly replayedTurns: number };
};

export type RepairRecord = { readonly schema: typeof REPAIR_SCHEMA; readonly root: string; readonly release: string; readonly round: number;
  readonly boundary: number; readonly status: 'admitted' | 'rejected'; readonly reason?: string; readonly kind?: 'repair' | 'continue';
  /** The whole repaired trajectory, each turn with its role. */
  readonly turns: readonly (CollectedTurn & { readonly role: 'student-prefix' | 'correction' })[] };

export type RepairSummary = { readonly records: string; readonly rows: string; readonly roots: number; readonly studentCorrect: number;
  readonly attempted: number; readonly admitted: number; readonly rejected: number; readonly skipped: readonly { root: string; reason: string }[] };

const sha = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const replyOf = (turn: CollectedTurn): ModelTurn => ({ calls: structuredClone(turn.reply.calls ?? []), text: turn.reply.text ?? '',
  ...(turn.reply.reasoning ? { reasoning: turn.reply.reasoning } : {}) }) as ModelTurn;
const opening = (turn: CollectedTurn) => openingText(turn.messages as Message[]);
const sameReply = (a: CollectedTurn['reply'], b: CollectedTurn['reply']) =>
  JSON.stringify([a.calls ?? [], a.text ?? '']) === JSON.stringify([b.calls ?? [], b.text ?? '']);

/** Call numbers by first opening, as replay.ts numbers recorded calls; null when two invocations share an opening
 * (indistinguishable openings cannot keep call ownership in a replay, as handoffs refuse them too). */
function callsOf(turns: readonly CollectedTurn[]): { numbers: number[]; openings: string[] } | null {
  const openings: string[] = [], owners = new Map<string, string | null>(), numbers: number[] = [];
  for (const turn of turns) {
    const text = opening(turn);
    if (!owners.has(text)) { owners.set(text, turn.call); openings.push(text); }
    else if (owners.get(text) !== turn.call) return null;
    numbers.push(openings.indexOf(text));
  }
  return { numbers, openings };
}

/**
 * Run the student on the training roots and repair what it gets wrong. `student` serves the release under `weights`
 * (strict); `teacher` is the authorized teacher. Records and admitted rows are appended to `records`/`rows`, and roots
 * already recorded for this release and round are skipped, so a rerun resumes.
 */
export async function runRepairs(options: { root: string; program: ProgramDescriptor; scaffold: WeightManifest; release: WeightManifest;
  weights: WeightBinding; student: ModelDriver; teacher: ModelDriver; teacherId: string; examples: readonly SplitExample[];
  round: number; records: string; rows: string; checkpointSha256: string; maxPerRoot?: number; maxBoundaries?: number;
  log?: (line: string) => void;
  /** Host services for one root (TuneOptions.services): the student runs in them, journaled; a repair replays that
   * journal and is rejected when its continuation needs a call the student's world never recorded. */
  services?: (example: SplitExample) => Services }): Promise<RepairSummary> {
  const maxPerRoot = options.maxPerRoot ?? 1, log = options.log ?? (() => {});
  mkdirSync(dirname(options.records), { recursive: true });
  const done = new Set((existsSync(options.records) ? readFileSync(options.records, 'utf8').split('\n').filter(Boolean) : [])
    .map(line => JSON.parse(line) as RepairRecord).filter(record => record.release === options.release.digest && record.round === options.round)
    .map(record => record.root));
  const roots = options.examples.filter(example => example.partition === 'train' && !example.protected && 'expected' in example);
  const actor = `student-stack:${options.checkpointSha256}:${options.release.digest}`;
  const skipped: { root: string; reason: string }[] = [];
  let studentCorrect = 0, attempted = 0, admitted = 0, rejected = 0;
  for (const example of roots) {
    if (done.has(example.id)) continue;
    // 1. The student's own run, served by the release (strict: no silent fallback).
    const run = await collect({ root: options.root, program: options.program, scaffold: options.release, examples: [example],
      partitions: ['train'], teacher: options.student, teacherId: actor, weights: options.weights, services: options.services });
    const world = run.roots[0]?.world;
    if (run.roots[0]?.oracle === true) { studentCorrect++; continue; }
    const student = run.turns;
    const calls = callsOf(student);
    if (!student.length || !calls) {
      const reason = !student.length ? 'the student made no model turn' : 'two invocations share an opening';
      skipped.push({ root: example.id, reason });
      appendFileSync(options.records, JSON.stringify({ schema: REPAIR_SCHEMA, root: example.id, release: options.release.digest,
        round: options.round, boundary: -1, status: 'rejected', reason, turns: [] } satisfies RepairRecord) + '\n');
      continue;
    }
    const nth = calls.numbers.map((call, index) => calls.numbers.slice(0, index).filter(other => other === call).length);
    const studentAt = (call: number, n: number) => student[calls.numbers.findIndex((other, index) => other === call && nth[index] === n)];
    let kept = 0, tried = 0;
    // 2. Boundaries from the latest backwards: the longest repairable student prefix.
    for (let boundary = student.length - 1; boundary >= 0 && kept < maxPerRoot; boundary--) {
      if (options.maxBoundaries !== undefined && tried >= options.maxBoundaries) break;
      tried++; attempted++;
      const script: ModelTurn[][] = [];
      student.slice(0, boundary).forEach((turn, index) => (script[calls.numbers[index]!] ??= []).push(replyOf(turn)));
      const teacher = options.teacher as (request: ModelTurnRequest, signal?: AbortSignal, turnOptions?: unknown) => Promise<ModelTurn>;
      const replay = scriptedDriver(script, calls.openings, request => teacher(request));
      const scripted = new Map<string, boolean>();
      const driver = Object.assign(async (request: ModelTurnRequest) => {
        const reply = await replay.driver(request);
        scripted.set(sha(request.messages), replay.places.at(-1)!.scripted);
        return reply;
      }, { contextWindow: (options.teacher as { contextWindow?: unknown }).contextWindow }) as unknown as ModelDriver;
      // The same program and inputs with the student's replies replayed re-execute the student's world.
      // The student's journaled world, replayed: no call acts on the live world again.
      const replayWorld = options.services && world ? replayServices(options.services(example), world) : undefined;
      const repaired = await collect({ root: options.root, program: options.program, scaffold: options.scaffold, examples: [example],
        partitions: ['train'], teacher: driver, teacherId: options.teacherId, ...(replayWorld ? { services: () => replayWorld.services } : {}) });
      const turns = repaired.turns.map(turn => ({ ...turn, role: scripted.get(sha(turn.messages)) ? 'student-prefix' as const : 'correction' as const }));
      const reject = (reason: string) => {
        rejected++;
        appendFileSync(options.records, JSON.stringify({ schema: REPAIR_SCHEMA, root: example.id, release: options.release.digest,
          round: options.round, boundary, status: 'rejected', reason, turns } satisfies RepairRecord) + '\n');
      };
      // 3. Exact verification: the replayed prefix and the boundary saw what the student saw; the root oracle holds.
      const places = replay.places;
      let mismatch: string | null = null, boundaryTurn: number | null = null;
      for (const [index, place] of places.entries()) {
        const original = studentAt(place.call, place.nth);
        const isBoundary = place.call === calls.numbers[boundary] && place.nth === nth[boundary];
        if (isBoundary && boundaryTurn === null) boundaryTurn = index;
        if (!place.scripted && !isBoundary) continue;
        const now = replay.turns[index]!.context as Message[];
        if (!original || JSON.stringify(observed(now)) !== JSON.stringify(observed(original.messages as Message[]))) {
          mismatch = `${place.scripted ? 'replayed' : 'boundary'} turn ${index} did not observe what the student observed`;
          break;
        }
      }
      const diverged = replayWorld?.divergence();
      if (diverged) { reject(`the continuation left the captured world: ${diverged}`); continue; }
      if (mismatch) { reject(mismatch); continue; }
      if (boundaryTurn === null) { reject('the replay never reached the boundary'); continue; }
      if (repaired.roots[0]?.oracle !== true) { reject(repaired.roots[0]?.error ? `the repaired run failed: ${repaired.roots[0].error.slice(0, 200)}` : 'the repaired run misses the root oracle'); continue; }
      const corrections = turns.filter(turn => turn.role === 'correction');
      if (corrections.some(turn => turn.reply.truncated)) { reject('a correction turn was truncated'); continue; }
      const boundaryReply = turns.find(turn => turn.role === 'correction' && opening(turn) === calls.openings[calls.numbers[boundary]!]);
      const kind = boundaryReply && sameReply(boundaryReply.reply, student[boundary]!.reply) ? 'continue' as const : 'repair' as const;
      const provenance = (turn: CollectedTurn): CorrectionProvenance => ({ round: options.round, kind,
        boundary: { index: boundary, call: student[boundary]!.call, turn: student[boundary]!.turn },
        student: { checkpointSha256: options.checkpointSha256, release: options.release.digest, actor, stack: student[boundary]!.stack ?? null },
        teacher: options.teacherId, call: turn.call,
        verification: { studentRootOracle: false, rootOracle: true, prefix: 'observed-identical', replayedTurns: turns.length - corrections.length } });
      // Rows: the teacher's turns only, with a repair-scoped call ID so they never collide with collection rows.
      const rows = corrections.filter(turn => turn.binding).map(({ role: _role, ...turn }) => ({ ...turn,
        call: `repair:r${options.round}:b${boundary}:${turn.call ?? 'root'}`, correction: provenance(turn) }));
      for (const row of rows) appendFileSync(options.rows, JSON.stringify(row) + '\n');
      appendFileSync(options.records, JSON.stringify({ schema: REPAIR_SCHEMA, root: example.id, release: options.release.digest,
        round: options.round, boundary, status: 'admitted', kind, turns } satisfies RepairRecord) + '\n');
      admitted++; kept++;
      log(`repair ${example.id} at turn ${boundary}: ${kind}, ${rows.length} correction rows`);
    }
    if (!kept) skipped.push({ root: example.id, reason: `no verified repair in ${tried} boundaries` });
  }
  return { records: options.records, rows: options.rows, roots: roots.length, studentCorrect, attempted, admitted, rejected, skipped };
}

/** Admitted correction rows, once each (an interrupted round may have appended a root's rows before its record). */
export function readRepairRows(path: string): CollectedTurn[] {
  const seen = new Set<string>();
  return (existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line) as CollectedTurn) : [])
    .filter(row => { const key = sha([row.call, row.messages, row.reply]); if (seen.has(key)) return false; seen.add(key); return true; });
}
