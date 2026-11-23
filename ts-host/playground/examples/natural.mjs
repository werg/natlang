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
export type Triage = { team: Team; priority: Priority; confidence: number };
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
  return { team: team.value, priority: priority.value, confidence: Math.min(team.confidence, priority.confidence) };
}
`,
    },
    inputs: { ticket: 'The app is down for all users.' },
    expected: { team: 'technical', priority: 'urgent', confidence: 1 } }),
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
];
