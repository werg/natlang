#!/usr/bin/env node
import {renderValue} from '../dist/native/agent.js';
/**
 * Convert natlang trajectory records to Neuralese form (src/compiler/neuralese-conversion.ts; DECISIONS.md 40–42):
 * reused texts (prompt pieces, program guidance, instructions used by several calls) become soft parameters, handover
 * notes become model writes read by later turns, large listing values become digest sites, single-use values stay
 * text, and every site is counted with its treatment.
 *
 *   node scripts/neuralese-convert-trajectories.mjs --out converted.jsonl --pieces pieces.jsonl [--summary s.json]
 *     [--instructions-reuse 2] [--instructions-share 0.1] [--limit N] input.jsonl [...]
 *
 * A first pass counts the distinct calls each instructions text serves and links child calls' returned values to the
 * caller outputs that print them, per collected run (child results). `pieces.jsonl` holds each soft parameter's
 * name, kind and initial text once (records name them only).
 */
import {createHash} from 'node:crypto';
import { createReadStream, createWriteStream, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { callOf, childCallIds, invocationOf, childFunctionNames, childReturn, convertTrajectory, instructionsDigest, NEURALESE_CONVERSION_VERSION, openingInstructions,
  printedResults } from '../dist/compiler/neuralese-conversion.js';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  'audit-only': { type: 'boolean', default: false }, out: { type: 'string' }, pieces: { type: 'string' }, summary: { type: 'string' }, limit: { type: 'string' },
  'instructions-reuse': { type: 'string' }, 'instructions-share': { type: 'string' },
} });
if ((!values['audit-only'] && (!values.out || !values.pieces)) || !positionals.length) {
  console.error('usage: neuralese-convert-trajectories.mjs --out FILE --pieces FILE [--summary FILE] [--instructions-reuse N] [--instructions-share F] [--limit N] input.jsonl...');
  process.exit(2);
}
const limit = values.limit ? Number(values.limit) : Infinity;
const instructionsReuse = values['instructions-reuse'] ? Number(values['instructions-reuse']) : 2;
const instructionsShare = values['instructions-share'] ? Number(values['instructions-share']) : 0.1;
// First pass: the distinct calls each instructions text serves (every turn of a call is its own record).
const callsByInstructions = new Map();
// Per run: child calls' returned values, and the caller eval outputs that show child results.
const returnedByRun = new Map(), outputsByRun = new Map(), producersByRun = new Map();
let counted = 0;
outer0: for (const input of positionals) {
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    if (counted++ >= limit) break outer0;
    let row;
    try { row = JSON.parse(line); } catch { continue; }
    if (!Array.isArray(row.messages)) continue;
    const run = callOf(row);
    const returned = childReturn(row);
    if (returned !== undefined) {
      (returnedByRun.get(run) ?? returnedByRun.set(run, new Set()).get(run)).add(returned);
      const producers = producersByRun.get(run) ?? producersByRun.set(run, []).get(run);
      const value = JSON.parse(row.target.tool_calls.find(c => c.function.name === 'return_result').function.arguments).value;
      const invocation = invocationOf(row), parent = row.source_ref?.parent_invocation_id;
      producers.push({id: String(row.id), invocation, value: returned, parent, renderings: [renderValue(value, {budget: Infinity}), JSON.stringify(value)]});
      // A structured result's text fields flow on their own (the caller takes \`found.facts\` into another call's
      // arguments): each long one is a producer too, written field by field.
      if (value && typeof value === 'object' && !Array.isArray(value)) for (const [field, text] of Object.entries(value))
        if (typeof text === 'string' && text.length >= 16 && text !== returned) {
          returnedByRun.get(run).add(text);
          producers.push({id: `${row.id}#${field}`, invocation, field, value: text, parent, renderings: [JSON.stringify(text).slice(1, -1)]});
        }
    }
    const children = childCallIds(row.messages, childFunctionNames(row));
    // A call's argument listing (scope_0) shows values its caller passed in, often another call's result.
    for (const message of row.messages) if (message.role === 'tool' && (children.has(String(message.tool_call_id)) || message.tool_call_id === 'scope_0') && typeof message.content === 'string')
      (outputsByRun.get(run) ?? outputsByRun.set(run, new Map()).get(run)).set(`${invocationOf(row)}:${message.tool_call_id}`, {invocation: invocationOf(row), text: message.content, argument: message.tool_call_id === 'scope_0'});
    const text = openingInstructions(row);
    if (text === undefined) continue;
    const digest = instructionsDigest(text);
    const calls = callsByInstructions.get(digest) ?? new Set();
    calls.add(`${callOf(row)}:${row.source_ref?.invocation_id ?? createHash('sha256').update(JSON.stringify(row.messages.slice(0,2))).digest('hex')}`);
    callsByInstructions.set(digest, calls);
  }
}
const instructionCalls = new Map([...callsByInstructions].map(([digest, calls]) => [digest, calls.size]));
const childResults = new Map();
for (const run of new Set([...returnedByRun.keys(), ...outputsByRun.keys()])) {
  const returned = [...returnedByRun.get(run) ?? []];
  const producers = producersByRun.get(run) ?? [];
  // Equal text is insufficient evidence to select one of several child invocations.
  // Keep those values exact, preserving usable data without inventing a recurrence edge.
  const readers = [];
  for (const output of outputsByRun.get(run)?.values() ?? []) {
    // An argument listing (scope_0) shows a passed value whole, as a quoted string; an eval output may print it anywhere.
    const shows = form => output.argument ? output.text.includes(`"${form}"`) : output.text.includes(form);
    for (const value of returned.filter(value => value.length >= 16 && [value,...(producers.find(p=>p.value===value)?.renderings??[])].some(shows))) {
      const all = producers.filter(p => p.value === value);
      // Parent links are observed runtime metadata. Without them, require a globally unique producer.
      const candidates = all.some(p => p.parent !== undefined) ? all.filter(p => p.parent === output.invocation) : all;
      if (candidates.length === 1 && candidates[0].invocation !== output.invocation &&
          !readers.some(r => r.invocation === output.invocation && r.value === value))
        readers.push({invocation: output.invocation, value, producer_id: candidates[0].id});
    }
  }
  const read = new Set(readers.map(r => r.value));
  childResults.set(run, { returned, read, producers, readers });
}
const out = values['audit-only'] ? null : createWriteStream(values.out, { flags: 'wx' });
const pieces = new Map();
const reuse = [...instructionCalls.values()];
const totals = { version: NEURALESE_CONVERSION_VERSION, instructions_reuse: instructionsReuse, instructions_share: instructionsShare,
  instruction_texts: { distinct: reuse.length, reused: reuse.filter(n => n >= instructionsReuse).length,
    calls_of_reused: reuse.filter(n => n >= instructionsReuse).reduce((sum, n) => sum + n, 0), calls: reuse.reduce((sum, n) => sum + n, 0) },
  records: 0, unreadable: 0, passed_through: 0, sites: {}, pieces: {} };
outer: for (const input of positionals) {
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    if (totals.records >= limit) break outer;
    totals.records++;
    let row;
    try { row = JSON.parse(line); } catch { totals.unreadable++; continue; }
    if (!Array.isArray(row.messages)) { totals.passed_through++; out?.write(line + '\n'); continue; }
    const { record, pieces: used } = convertTrajectory(row, { instructionCalls, instructionsReuse, instructionsShare, childResults });
    for (const piece of used) if (!pieces.has(piece.name)) pieces.set(piece.name, piece);
    for (const [kind, site] of Object.entries(record.neuralese_conversion.sites)) {
      const total = totals.sites[kind] ??= { converted: 0, exact: {} };
      total.converted += site.converted;
      for (const [reason, n] of Object.entries(site.exact)) total.exact[reason] = (total.exact[reason] ?? 0) + n;
    }
    if (out && !out.write(JSON.stringify(record) + '\n')) await new Promise(resolve => out.once('drain', resolve));
  }
}
if (out) await new Promise(resolve => out.end(resolve));
if (!values['audit-only']) writeFileSync(values.pieces, [...pieces.values()].map(piece => JSON.stringify(piece)).join('\n') + '\n', { flag: 'wx' });
for (const piece of pieces.values()) totals.pieces[piece.kind] = (totals.pieces[piece.kind] ?? 0) + 1;
const summary = JSON.stringify(totals, null, 2);
if (values.summary) writeFileSync(values.summary, summary + '\n');
console.log(summary);
