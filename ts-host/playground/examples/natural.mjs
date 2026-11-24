import { natural } from './builders.mjs';

/** A TypeScript-orchestrated semantic-FP playground example (runs in browser with `@natlang/browser` and `@natlang/browser/std`). */
const semanticProgram = ({ id, name, category = 'Semantic FP', level, description, concepts, root, files, inputs = {}, expected }) => ({
  id, name, category, level, description, concepts, modelRequired: true, root, files, inputs, expected,
});

export const naturalExamples = [
  natural({ id: 'natural-seven', name: 'Return seven', level: 'Beginner',
    description: 'Produce a typed scalar from a natural instruction.',
    concepts: ['typed output'], root: 'tasks/answer.nl',
    source: `---
description: Produce one typed answer from natural instructions.
returns: number
---
Return the number 7.`, expected: 7 }),
  natural({ id: 'feedback-tone', name: 'Feedback tone', category: 'Classification', level: 'Beginner',
    description: 'Classify outside customer text with a calibrated decision readout.',
    concepts: ['readout: decision', 'Untrusted', 'literal union'], root: 'feedback/tone.nl',
    source: `---
description: Classify the tone of an external customer comment.
args:
  comment: Untrusted<string>
returns: Tone
readout: decision
types:
  Tone: '"positive" | "negative" | "neutral"'
---
Choose the tone that best fits comment.`,
    inputs: { comment: 'The setup was easy and the team was helpful.' }, expected: 'positive' }),
  natural({ id: 'extract-contact', name: 'Extract contact', category: 'Extraction', level: 'Beginner',
    description: 'Extract a name and email from untrusted text into a typed record.',
    concepts: ['extraction', 'typed records', 'Untrusted'], root: 'contacts/extract.nl',
    source: `---
description: Extract a contact record from an incoming message.
args:
  message: Untrusted<string>
returns: Contact
types:
  Contact: '{ name: string, email: string }'
---
Read message and find the person's full name and email address.
Copy both verbatim as written, without inventing missing details.`,
    inputs: { message: 'Please contact Maya Chen at maya@example.com.' },
    expected: { name: 'Maya Chen', email: 'maya@example.com' } }),
  natural({ id: 'issue-fields', name: 'Issue fields', category: 'Extraction', level: 'Intermediate',
    description: 'Extract structured fields with a narrow literal union for urgency.',
    concepts: ['extraction', 'literal union', 'Untrusted'], root: 'issues/fields.nl',
    source: `---
description: Extract structured issue metadata from an external support request.
args:
  message: Untrusted<string>
returns: Issue
types:
  Issue: '{ area: string, urgency: "high" | "normal", customer: string }'
---
Extract customer, affected area, and urgency from message.
Choose urgency "high" only for an active outage or data loss; otherwise "normal".`,
    inputs: { message: 'From Northwind: checkout is down for every customer right now.' },
    expected: { area: 'checkout', urgency: 'high', customer: 'Northwind' } }),
  natural({ id: 'policy-decision', name: 'Policy decision', category: 'Decisions', level: 'Intermediate',
    description: 'Score a finite policy verdict with calibrated probabilities and a grounded reason.',
    concepts: ['readout: decision', 'structured decision', 'Untrusted'], root: 'policy/decide.nl',
    source: `---
description: Decide whether a request is allowed under a rule, and whether it needs human review.
args:
  request: Untrusted<string>
  rule: string
returns: Decision
readout: decision
types:
  Decision: '{ allowed: boolean, needsReview: boolean }'
---
Apply rule to request. Set allowed to true when rule permits request, and needsReview to true only when request
leaves a key condition ambiguous or unstated.`,
    inputs: { request: 'Refund a purchase made 10 days ago.', rule: 'Refunds are allowed within 30 days.' },
    expected: { allowed: true, needsReview: false } }),
  natural({ id: 'meeting-actions', name: 'Meeting actions', category: 'Extraction', level: 'Intermediate',
    description: 'Extract ordered action items from untrusted meeting notes.',
    concepts: ['extraction', 'arrays', 'Untrusted'], root: 'meetings/actions.nl',
    source: `---
description: List explicit next actions from meeting notes in order.
args:
  notes: Untrusted<string>
returns: string[]
---
List only the explicit next actions committed to in notes, in the order they appear.`,
    inputs: { notes: 'Maya will draft the proposal. We discussed pricing. Leo will send the test report.' },
    expected: ['Maya will draft the proposal.', 'Leo will send the test report.'] }),
  natural({ id: 'sql-risk', name: 'SQL risk triage', category: 'Classification', level: 'Intermediate',
    description: 'Triage untrusted query input with a calibrated decision readout.',
    concepts: ['security triage', 'readout: decision', 'Untrusted'], root: 'security/sql_risk.nl',
    source: `---
description: Decide whether an external query fragment attempts SQL injection.
args:
  query: Untrusted<string>
returns: Risk
readout: decision
types:
  Risk: '"safe" | "suspicious"'
---
Choose "suspicious" when query attempts to alter SQL syntax or bypass predicates; otherwise "safe".`,
    inputs: { query: "' OR 1=1 --" }, expected: 'suspicious' }),
  natural({ id: 'evidence-summary', name: 'Evidence summary', category: 'Synthesis', level: 'Intermediate',
    description: 'Answer a claim grounded strictly in supplied evidence, refined by type.',
    concepts: ['grounding', 'Is<T, P>', 'Untrusted'], root: 'research/summarize.nl',
    source: `---
description: Answer a claim from evidence in one grounded sentence.
args:
  claim: string
  evidence: Untrusted<string>[]
returns: Answer
types:
  Answer: 'Is<string, "one concise sentence grounded strictly in the supplied evidence">'
---
Answer claim using only evidence. If evidence is insufficient, say so plainly.`,
    inputs: { claim: 'Did the service recover?', evidence: ['Errors fell to zero at 14:05 UTC.', 'The health check passed at 14:07 UTC.'] },
    expected: 'The service recovered by 14:07 UTC, when the health check passed after errors fell to zero.' }),
  semanticProgram({ id: 'support-triage', name: 'Support triage', category: 'Semantic FP',
    level: 'Advanced', description: 'TypeScript orchestrates two calibrated decision functions in parallel with exact assembly.',
    concepts: ['TS orchestration', 'readout: decision', 'decide', 'Untrusted'], root: 'support/main.ts',
    files: {
      'support/types.ts': `export type Team = 'billing' | 'technical';
export type Priority = 'urgent' | 'normal';
export type Triage = { team: Team; priority: Priority; settled: boolean };
`,
      'support/classify.nl': `---
description: Route a support ticket to billing or technical.
args:
  ticket: Untrusted<string>
returns: Team
readout: decision
---
Choose "billing" for payment or invoice problems and "technical" for product or outage failures.
`,
      'support/prioritize.nl': `---
description: Decide whether a ticket is an active urgent outage.
args:
  ticket: Untrusted<string>
returns: Priority
readout: decision
---
Choose "urgent" for an active outage or data loss affecting users now; otherwise "normal".
`,
      'support/main.ts': `import { decide, untrusted } from '@natlang/browser';
import classify from './classify.nl';
import prioritize from './prioritize.nl';
import type { Triage } from './types.js';

export async function main(input: { ticket: string }): Promise<Triage> {
  const safe = untrusted(input.ticket, 'incoming ticket');
  const [team, priority] = await Promise.all([decide(classify, safe), decide(prioritize, safe)]);
  // settled: both decisions are confident enough to act on without a person (or a stronger model) looking.
  return { team: team.value, priority: priority.value, settled: Math.min(team.confidence, priority.confidence) >= 0.8 };
}
`,
    },
    inputs: { ticket: 'The app is down for all users.' },
    expected: { team: 'technical', priority: 'urgent', settled: true } }),
  semanticProgram({ id: 'incident-brief', name: 'Incident brief', category: 'Semantic FP',
    level: 'Advanced', description: 'Combine a calibrated severity decision with an inline refined headline.',
    concepts: ['inline nl', 'Is<T, P>', 'readout: decision'], root: 'incident/main.ts',
    files: {
      'incident/types.ts': `export type Severity = 'critical' | 'minor';
export type Headline = Is<string, 'one factual sentence naming the user impact'>;
export type Brief = { severity: Severity; headline: Headline };
`,
      'incident/severity.nl': `---
description: Judge incident severity from a log excerpt.
args:
  log: Untrusted<string>
returns: Severity
readout: decision
---
Choose "critical" when the service is unavailable to users or losing data; otherwise "minor".
`,
      'incident/main.ts': `import { nl, untrusted } from '@natlang/browser';
import severity from './severity.nl';
import type { Brief, Headline } from './types.js';

export async function main(input: { log: string }): Promise<Brief> {
  const safe = untrusted(input.log, 'incident log');
  const headline = nl<(log: Untrusted<string>) => Headline>\`Write a factual one-sentence incident headline from log.\`;
  const [sev, line] = await Promise.all([severity(safe), headline(safe)]);
  return { severity: sev, headline: line };
}
`,
    },
    inputs: { log: 'Checkout was unavailable to all users for 12 minutes.' },
    expected: { severity: 'critical', headline: 'Checkout was unavailable to all users for 12 minutes.' } }),
  semanticProgram({ id: 'release-review', name: 'Release review', category: 'Semantic FP',
    level: 'Advanced', description: 'Exact TypeScript gate paired with semanticRepair over a concise release summary.',
    concepts: ['@natlang/std', 'semanticRepair', 'inline nl'], root: 'release/main.ts',
    files: {
      'release/types.ts': `export type ReleaseReview = { ship: boolean; summary: string };
`,
      'release/main.ts': `import { nl, untrusted } from '@natlang/browser';
import { semanticRepair } from '@natlang/browser/std';
import type { ReleaseReview } from './types.js';

const words = (text: string) => text.trim().split(/\\s+/).filter(Boolean).length;

export async function main(input: { notes: string; tests_passed: boolean }): Promise<ReleaseReview> {
  const safeNotes = untrusted(input.notes, 'release notes');
  const draft: string = await nl\`Summarize safeNotes in one user-facing sentence.\`(safeNotes);
  const fix = nl<(text: string, problems: string[]) => string>\`Shorten text so none of problems applies while keeping all facts.\`;
  const repaired = await semanticRepair(draft, {
    check: text => words(text) <= 18 ? [] : [\`Summary has \${words(text)} words; keep it under 19 words.\`],
    fix,
    attempts: 2,
  });
  return { ship: input.tests_passed, summary: repaired.value };
}
`,
    },
    inputs: { notes: 'The search page is faster and keyboard navigation is improved.', tests_passed: true },
    expected: { ship: true, summary: 'Search is faster and keyboard navigation is improved.' } }),
  semanticProgram({ id: 'survey-themes', name: 'Survey themes', category: 'Semantic FP',
    level: 'Advanced', description: 'Partition feedback with @natlang/browser/std semanticPartition and synthesize the top theme.',
    concepts: ['@natlang/std', 'semanticPartition', 'readout: decision'], root: 'survey/main.ts',
    files: {
      'survey/types.ts': `export type Theme = 'speed' | 'usability' | 'other';
export type ThemeReport = { counts: Record<string, number>; summary: string };
`,
      'survey/label.nl': `---
description: Classify a survey comment into a product theme.
args:
  comment: Untrusted<string>
returns: Theme
readout: decision
---
Choose "speed" for performance complaints, "usability" for interface complaints, or "other".
`,
      'survey/main.ts': `import { nl, untrusted } from '@natlang/browser';
import { semanticPartition } from '@natlang/browser/std';
import label from './label.nl';
import type { ThemeReport } from './types.js';

export async function main(input: { comments: string[] }): Promise<ThemeReport> {
  const safe = input.comments.map((text, i) => untrusted(text, \`comment #\${i + 1}\`));
  const { groups } = await semanticPartition(safe, label);
  const counts = Object.fromEntries(Object.entries(groups).map(([k, items]) => [k, items.length]));
  const summary: string = await nl\`Write one sentence naming the dominant complaint theme given counts and groups.\`();
  return { counts, summary };
}
`,
    },
    inputs: { comments: ['The page is slow.', 'Search takes forever.', 'The buttons are confusing.'] },
    expected: { counts: { speed: 2, usability: 1 }, summary: 'Speed is the most common issue.' } }),
  semanticProgram({ id: 'policy-lenses', name: 'Policy lenses (.in)', category: 'Semantic FP',
    level: 'Advanced', description: 'Bind one reusable verdict function to two different written policies using withContext.',
    concepts: ['@natlang/std', 'withContext', '.in(context)', 'readout: decision'], root: 'lenses/main.ts',
    files: {
      'lenses/types.ts': `export type Verdict = 'allow' | 'deny';
export type Comparison = { strict: Verdict; generous: Verdict };
`,
      'lenses/verdict.nl': `---
description: Decide whether a customer request is allowed under the bound policy.
args:
  request: Untrusted<string>
returns: Verdict
readout: decision
---
Judge request under policy (the written policy in this context). Answer "allow" when policy permits it, else "deny".
`,
      'lenses/main.ts': `import { untrusted } from '@natlang/browser';
import { withContext } from '@natlang/browser/std';
import verdict from './verdict.nl';
import type { Comparison } from './types.js';

export async function main(input: { request: string }): Promise<Comparison> {
  const safe = untrusted(input.request, 'customer request');
  const strict = withContext(verdict, { policy: 'Refunds only within 14 days for unopened items.' });
  const generous = withContext(verdict, { policy: 'Refunds within 60 days even if opened.' });
  const [strictResult, generousResult] = await Promise.all([strict(safe), generous(safe)]);
  return { strict: strictResult, generous: generousResult };
}
`,
    },
    inputs: { request: 'I opened the backpack 20 days ago and would like a refund.' },
    expected: { strict: 'deny', generous: 'allow' } }),
  semanticProgram({ id: 'inbox-planner', name: 'Inbox planner', category: 'Semantic FP',
    level: 'Beginner', description: 'The natlang init default: route every message with a calibrated decision and draft replies only where needed.',
    concepts: ['readout: decision', 'decide', 'inline nl', 'Untrusted', 'Is<T, P>'], root: 'inbox/main.ts',
    files: {
      'inbox/types.ts': `export type Message = { from: string; subject: string; body: Untrusted<string> };
export type Route = 'reply' | 'fyi' | 'ignore';
export type Reply = Is<string, 'a polite reply of at most three sentences that makes no commitment the message does not ask for'>;
export type Plan = { from: string; subject: string; route: Route; confidence: number; reply?: Reply };
`,
      'inbox/route.nl': `---
description: Decide what a message needs from its reader.
args:
  message: Message
returns: Route
readout: decision
---
reply when the sender asks the reader a question or asks them to do something; fyi when a person or team tells the
reader something worth knowing and asks nothing; ignore when it is advertising, a newsletter or an automated message
the reader does not need.
`,
      'inbox/main.ts': `import { decide, nl, untrusted } from '@natlang/browser';
import route from './route.nl';
import type { Message, Plan, Reply } from './types.js';

export async function main(input: { messages: { from: string; subject: string; body: string }[] }): Promise<Plan[]> {
  const messages: Message[] = input.messages.map(m => ({ ...m, body: untrusted(m.body, \`message from \${m.from}\`) }));
  const decisions = await Promise.all(messages.map(message => decide(route, message)));
  const draft = nl<(message: Message) => Reply>\`Draft a reply to message for its reader.\`;
  return Promise.all(messages.map(async (message, i) => {
    const { value, confidence } = decisions[i]!;
    const planned: Plan = { from: message.from, subject: message.subject, route: value, confidence };
    if (value === 'reply') planned.reply = await draft(message);
    return planned;
  }));
}
`,
    },
    inputs: { messages: [
      { from: 'lee@example.com', subject: 'New designer', body: 'Hi team, Maria joins us on Monday as our new designer.' },
      { from: 'deals@shop.example', subject: '50% off this weekend', body: 'Our biggest sale of the year starts now. Click to shop.' },
    ] },
    expected: [
      { from: 'lee@example.com', subject: 'New designer', route: 'fyi', confidence: 0.9 },
      { from: 'deals@shop.example', subject: '50% off this weekend', route: 'ignore', confidence: 0.9 },
    ] }),
  semanticProgram({ id: 'calibrated-gate', name: 'Calibrated gate (escalate)', category: 'Semantic FP',
    level: 'Intermediate', description: 'Gate a local decision with @natlang/browser/std escalate using both a confidence floor and an asymmetric risk floor.',
    concepts: ['@natlang/std', 'escalate', 'readout: decision', 'Untrusted'], root: 'gate/main.ts',
    files: {
      'gate/types.ts': `export type Verdict = 'approve' | 'review' | 'block';
export type GateResult = { verdict: Verdict; confidence: number; escalated: boolean; reason?: string };
`,
      'gate/verdict.nl': `---
description: Decide whether a proposed change can be approved, needs review, or should be blocked.
args:
  change: Untrusted<string>
returns: Verdict
readout: decision
---
Choose "block" for security regressions or data loss, "review" for risky logic edits, and "approve" for safe routine updates.
`,
      'gate/main.ts': `import { untrusted } from '@natlang/browser';
import { escalate } from '@natlang/browser/std';
import verdict from './verdict.nl';
import type { GateResult } from './types.js';

export async function main(input: { change: string }): Promise<GateResult> {
  const safe = untrusted(input.change, 'proposed change');
  // Escalate when local confidence < 0.85 OR when P(block) >= 0.25.
  // Without a configured 'teacher' model, escalate keeps the local answer with escalated: false and reason: 'unavailable:...'.
  const decideChange = escalate(verdict, { to: 'teacher', floor: 0.85, risk: { block: 0.25 } });
  const result = await decideChange(safe);
  return {
    verdict: result.value,
    confidence: result.confidence,
    escalated: result.escalated,
    ...(result.reason ? { reason: result.reason } : {}),
  };
}
`,
    },
    inputs: { change: 'Update the README typo in the installation section.' },
    expected: { verdict: 'approve', confidence: 0.95, escalated: false } }),
  semanticProgram({ id: 'pluggable-policy', name: 'Rule or judgment (pluggable)', category: 'Semantic FP',
    level: 'Intermediate', description: 'Put a fast crisp rule and a typed natural-language judgment behind one pluggable switch supporting shadow mode.',
    concepts: ['pluggable', 'shadow mode', 'readout: decision', 'Untrusted'], root: 'urgent/main.ts',
    files: {
      'urgent/types.ts': `export type Mode = 'crisp' | 'nl' | 'shadow';
export type UrgencyReport = { mode: Mode; urgentIds: string[] };
`,
      'urgent/is_urgent.nl': `---
description: Decide whether an incoming ticket describes an urgent outage or blocking failure.
args:
  ticket: Untrusted<string>
returns: boolean
readout: decision
---
Return true when ticket reports an active outage, security incident, or data loss affecting users now; otherwise false.
`,
      'urgent/main.ts': `import { pluggable, untrusted, type PluggableMode } from '@natlang/browser';
import isUrgentNl from './is_urgent.nl';
import type { Mode, UrgencyReport } from './types.js';

const isUrgentCrisp = (ticket: Untrusted<string>): boolean => /\\b(outage|down|fire|data loss)\\b/i.test(String(ticket));

export async function main(input: { mode: Mode; tickets: { id: string; text: string }[] }): Promise<UrgencyReport> {
  const check = pluggable<[Untrusted<string>], boolean>(
    { crisp: isUrgentCrisp, nl: ticket => isUrgentNl(ticket) },
    input.mode as PluggableMode,
    { name: 'urgent.check', default: 'nl', serve: 'crisp' },
  );
  const flags = await Promise.all(input.tickets.map(t => check(untrusted(t.text, \`ticket \${t.id}\`))));
  return { mode: input.mode, urgentIds: input.tickets.filter((_, i) => flags[i]).map(t => t.id) };
}
`,
    },
    inputs: { mode: 'shadow', tickets: [
      { id: 't1', text: 'Checkout is down for every customer.' },
      { id: 't2', text: 'Please update our billing address next week.' },
    ] },
    expected: { mode: 'shadow', urgentIds: ['t1'] } }),
  semanticProgram({ id: 'semantic-join-rank', name: 'Match & rank (std)', category: 'Semantic FP',
    level: 'Advanced', description: 'Pair records with exact blocking plus semanticMatch, then order matches with debiased pairwise semanticRank.',
    concepts: ['@natlang/std', 'semanticMatch', 'semanticRank', 'readout: decision'], root: 'match/main.ts',
    files: {
      'match/types.ts': `export type Incident = { id: string; service: string; detail: string };
export type Runbook = { id: string; service: string; title: string };
export type MatchSummary = { matched: { incidentId: string; runbookId: string }[]; topIncidentIds: string[] };
`,
      'match/fits.nl': `---
description: Decide whether a runbook addresses an incident.
args:
  detail: Untrusted<string>
  runbookTitle: string
returns: boolean
readout: decision
---
Return true when runbookTitle directly addresses the failure described in detail; otherwise false.
`,
      'match/main.ts': `import { untrusted } from '@natlang/browser';
import { semanticMatch, semanticRank } from '@natlang/browser/std';
import fits from './fits.nl';
import type { Incident, MatchSummary, Runbook } from './types.js';

export async function main(input: { incidents: Incident[]; runbooks: Runbook[] }): Promise<MatchSummary> {
  // Exact blocking restricts semantic comparisons to matching services.
  const { pairs } = await semanticMatch(input.incidents, input.runbooks, {
    block: (inc, rb) => inc.service === rb.service,
    match: (inc, rb) => fits(untrusted(inc.detail, \`incident \${inc.id}\`), rb.title),
  });
  const ranked = await semanticRank(input.incidents, 'highest immediate customer impact to investigate first');
  return {
    matched: pairs.map(p => ({ incidentId: p.left.id, runbookId: p.right.id })),
    topIncidentIds: ranked.ranked.map(inc => inc.id),
  };
}
`,
    },
    inputs: {
      incidents: [
        { id: 'i1', service: 'payments', detail: 'Card charges are timing out across all regions.' },
        { id: 'i2', service: 'search', detail: 'Autocomplete latency rose by 40ms.' },
      ],
      runbooks: [
        { id: 'r1', service: 'payments', title: 'Recover payment gateway timeouts' },
        { id: 'r2', service: 'search', title: 'Rebuild search synonym cache' },
      ],
    },
    expected: { matched: [{ incidentId: 'i1', runbookId: 'r1' }], topIncidentIds: ['i1', 'i2'] } }),
  semanticProgram({ id: 'policy-compiler', name: 'Policy compiler (compileRule)', category: 'Semantic FP',
    level: 'Advanced', description: 'Synthesize and example-verify a reusable judgment function from a plain-language rule using compileRule.',
    concepts: ['@natlang/std', 'compileRule', 'higher-order', 'inline nl'], root: 'compiler/main.ts',
    files: {
      'compiler/types.ts': `export type PolicyReport = { allowed: string[]; summary: string };
`,
      'compiler/main.ts': `import { nl, untrusted, type Untrusted } from '@natlang/browser';
import { compileRule } from '@natlang/browser/std';
import type { PolicyReport } from './types.js';

export async function main(input: { rule: string; items: string[] }): Promise<PolicyReport> {
  const check = await compileRule<Untrusted<string>, boolean>(input.rule, {
    examples: [
      { name: 'approved travel', input: untrusted('Economy train ticket for client visit', 'example 1'), expected: true },
      { name: 'personal purchase', input: untrusted('Personal video game console', 'example 2'), expected: false },
    ],
    attempts: 2,
  });
  const safe = input.items.map((text, i) => untrusted(text, \`item \${i + 1}\`));
  const flags = await Promise.all(safe.map(item => check(item)));
  const allowed = input.items.filter((_, i) => flags[i]);
  const summary: string = await nl\`Write one brief sentence summarizing which expenses in allowed passed the policy.\`(allowed);
  return { allowed, summary };
}
`,
    },
    inputs: {
      rule: 'Approve reasonable work travel and software expenses; reject personal purchases.',
      items: ['Hotel room for engineering conference', 'Home espresso machine'],
    },
    expected: { allowed: ['Hotel room for engineering conference'], summary: 'Only the conference hotel room passed the expense policy.' } }),
  semanticProgram({ id: 'soft-summary', name: 'Soft-state reduce (Neuralese)', category: 'Semantic FP',
    level: 'Advanced', description: 'Tree-reduce sections with semanticReduce({ carry: "neuralese" }) so intermediate views stay compressed until final readout.',
    concepts: ['@natlang/std', 'semanticReduce', 'carry: neuralese', 'view'], root: 'digest/main.ts',
    files: {
      'digest/types.ts': `export type DigestReport = { sectionCount: number; summary: string };
`,
      'digest/summarize.nl': `---
description: Summarize one document section into a brief running note.
args:
  prior: string
  section: Untrusted<string>
returns: string
---
Incorporate the key facts and decisions from section into prior, keeping the note concise.
`,
      'digest/combine.nl': `---
description: Combine two partial section summaries into one concise synthesis.
args:
  left: string
  right: string
returns: string
---
Merge left and right into one coherent, concise summary keeping every concrete decision and risk.
`,
      'digest/main.ts': `import { untrusted } from '@natlang/browser';
import { semanticReduce } from '@natlang/browser/std';
import summarize from './summarize.nl';
import combine from './combine.nl';
import type { DigestReport } from './types.js';

export async function main(input: { sections: string[] }): Promise<DigestReport> {
  const safe = input.sections.map((text, i) => untrusted(text, \`section \${i + 1}\`));
  // On a Neuralese-capable runtime, carry: 'neuralese' routes intermediate states through builtin view() and reads once at the end.
  const summary = await semanticReduce(safe, '', (prior, section) => summarize(prior, section), {
    mode: 'tree',
    combine: (left, right) => combine(left, right),
    carry: 'neuralese',
    viewInstructions: 'Preserve concrete decisions, dates, and risks.',
  });
  return { sectionCount: input.sections.length, summary };
}
`,
    },
    inputs: { sections: [
      'Q3 launch moved to October 14 after load testing passed.',
      'Billing migration requires a 10-minute read-only window on October 12.',
    ] },
    expected: { sectionCount: 2, summary: 'The Q3 launch is set for October 14 after load testing passed, following a 10-minute read-only billing migration window on October 12.' } }),
];
