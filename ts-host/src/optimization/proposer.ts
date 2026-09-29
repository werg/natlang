import type { Candidate, ComponentValue } from '../adaptation/types.js';
import { canonical, fingerprint } from '../adaptation/identity.js';
import { parseStrictJSON } from '../adaptation/schema.js';
import { validateCandidate } from '../adaptation/compatibility.js';
import type { ProposalContext } from './types.js';
/** The search wire codec is plain text or JSON segments; JS expressions remain compiler-owned. */
export function encodeCandidate(candidate: Candidate): Readonly<Record<string, string>> {
  return Object.fromEntries(Object.entries(candidate).map(([key, value]) => [key,
    value.kind === 'program.guidance' ? value.text : value.template.slotIds.length ? canonical(value.template.segments) : value.template.segments[0]!]));
}
export function decodeProposal(encoded: unknown, context: Pick<ProposalContext, 'candidate' | 'components' | 'keys'>): Candidate {
  if (!encoded || typeof encoded !== 'object' || Array.isArray(encoded)) throw new Error('proposal must be an object');
  const entries = Object.entries(encoded);
  if (entries.length !== context.keys.length || entries.some(([key, text]) => !context.keys.includes(key) || typeof text !== 'string')) throw new Error('proposal must include exactly the selected update keys as strings');
  const result: Record<string, ComponentValue> = { ...context.candidate };
  for (const [key, text] of entries) {
    const current = context.candidate[key]!;
    if (current.kind === 'program.guidance') result[key] = { kind: current.kind, text: text as string };
    else {
      const segments = current.template.slotIds.length ? parseStrictJSON(text as string) : [text];
      if (!Array.isArray(segments) || segments.some(segment => typeof segment !== 'string')) throw new Error('interpolated instructions require JSON static segment strings');
      result[key] = { kind: current.kind, template: { slotIds: current.template.slotIds, segments: segments as string[] } };
    }
  }
  return validateCandidate(result, context.components);
}
export async function propose(context: ProposalContext): Promise<Candidate> {
  const encoded = encodeCandidate(context.candidate);
  const selected = context.components.filter(component => context.keys.includes(component.key));
  let error = '';
  for (let repair = 0; repair <= context.maxRepairs; repair++) {
    const proposalSeed = context.seed === undefined ? null : parseInt(fingerprint({ seed: context.seed, keys: context.keys, repair }).slice(0, 8), 16) >>> 0;
    const turn = await context.gateway.request(context.driver, { seed: proposalSeed, max_tokens: 4096, tools: [], messages: [
      { role: 'system', content: 'Improve natural-language instructions using training feedback. Return ONLY a JSON object mapping each requested component key to its replacement string. Keep immutable contracts and capture names. For interpolated templates each string encodes a JSON array of static segments of the original length. Do not generate JavaScript, slots, signatures, or tools.' },
      { role: 'user', content: canonical({ components: selected, current: Object.fromEntries(context.keys.map(key => [key, encoded[key]])),
        feedback: context.feedback.slice(0, 12), validationError: error }).slice(0, 64000) } ] }, context.signal, 'reflection');
    try { if (typeof turn.text !== 'string') throw new Error('reflection model returned no structured text'); return decodeProposal(parseStrictJSON(turn.text), context); }
    catch (invalid) { error = invalid instanceof Error ? invalid.message : String(invalid); if (repair === context.maxRepairs) throw new Error('invalid proposal: ' + error); }
  }
  throw new Error('proposal repair exhausted');
}
