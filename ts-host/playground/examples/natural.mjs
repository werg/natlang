import { natural } from './builders.mjs';

export const naturalExamples = [
  natural({ id: 'natural-seven', name: 'Return seven', level: 'Beginner',
    description: 'Produce a typed scalar from a natural instruction.',
    concepts: ['write', 'typed output'], root: 'tasks/answer.nl',
    source: `---
description: Produce one typed answer from natural instructions.
returns: Num
---
Write the number 7 to return.`, expected: 7 }),
  natural({ id: 'feedback-tone', name: 'Feedback tone', category: 'Classification', level: 'Beginner',
    description: 'Classify a customer comment into a closed set of labels.',
    concepts: ['classification', 'enum'], root: 'feedback/tone.nl',
    source: `---
args:
  comment: Text
returns: Tone
types:
  Tone: '"positive" | "negative" | "neutral"'
---
Read args/comment. Write the best matching tone to return.`,
    inputs: { comment: 'The setup was easy and the team was helpful.' }, expected: 'positive' }),
  natural({ id: 'extract-contact', name: 'Extract contact', category: 'Extraction', level: 'Beginner',
    description: 'Extract a name and email into a typed record.',
    concepts: ['extraction', 'typed records'], root: 'contacts/extract.nl',
    source: `---
args:
  message: Text
returns: Contact
types:
  Contact: '{ name: Text, email: Text }'
---
Read args/message and find the person's full name and email address.
Copy both exactly as written, without inventing missing details.
Write a Contact record with name and email to return.`,
    inputs: { message: 'Please contact Maya Chen at maya@example.com.' },
    expected: { name: 'Maya Chen', email: 'maya@example.com' } }),
  natural({ id: 'issue-fields', name: 'Issue fields', category: 'Extraction', level: 'Intermediate',
    description: 'Read a support request and produce a structured issue record.',
    concepts: ['extraction', 'classification'], root: 'issues/fields.nl',
    source: `---
args:
  message: Text
returns: Issue
types:
  Issue: '{ area: Text, urgency: Text, customer: Text }'
---
Read args/message. Extract the customer, affected area, and urgency. Use urgency "high" only for a current outage; otherwise use "normal". Write the record to return.`,
    inputs: { message: 'From Northwind: checkout is down for every customer right now.' },
    expected: { area: 'checkout', urgency: 'high', customer: 'Northwind' } }),
  natural({ id: 'policy-decision', name: 'Policy decision', category: 'Decisions', level: 'Intermediate',
    description: 'Choose a decision and cite the applicable rule in a typed output.',
    concepts: ['policy', 'explanation'], root: 'policy/decide.nl',
    source: `---
args:
  request: Text
  rule: Text
returns: Decision
types:
  Decision: '{ allowed: Bool, reason: Text }'
---
Apply args/rule to args/request. Write allowed and a brief reason grounded in the rule to return.`,
    inputs: { request: 'Refund a purchase made 10 days ago.', rule: 'Refunds are allowed within 30 days.' },
    expected: { allowed: true, reason: 'The purchase is within the 30-day refund window.' } }),
  natural({ id: 'meeting-actions', name: 'Meeting actions', category: 'Extraction', level: 'Intermediate',
    description: 'Extract ordered action items from meeting notes.',
    concepts: ['extraction', 'arrays'], root: 'meetings/actions.nl',
    source: `---
args:
  notes: Text
returns: Text[]
---
List only the explicit next actions in args/notes, in the order they appear. Write the list to return.`,
    inputs: { notes: 'Maya will draft the proposal. We discussed pricing. Leo will send the test report.' },
    expected: ['Maya will draft the proposal.', 'Leo will send the test report.'] }),
  natural({ id: 'sql-risk', name: 'SQL risk triage', category: 'Classification', level: 'Intermediate',
    description: 'Classify a suspicious query fragment using a closed risk label.',
    concepts: ['security triage', 'classification'], root: 'security/sql_risk.nl',
    source: `---
args:
  query: Text
returns: Risk
types:
  Risk: '"safe" | "suspicious"'
---
Classify args/query as suspicious when it contains an attempt to change query meaning through injected SQL syntax. Otherwise classify it as safe. Write the label to return.`,
    inputs: { query: "' OR 1=1 --" }, expected: 'suspicious' }),
  natural({ id: 'evidence-summary', name: 'Evidence summary', category: 'Synthesis', level: 'Intermediate',
    description: 'Summarize a claim using supplied evidence only.',
    concepts: ['grounding', 'synthesis'], root: 'research/summarize.nl',
    source: `---
args:
  claim: Text
  evidence: Text[]
returns: Text
---
Answer args/claim using only args/evidence. If the evidence is insufficient, say so. Write one concise sentence to return.`,
    inputs: { claim: 'Did the service recover?', evidence: ['Errors fell to zero at 14:05 UTC.', 'The health check passed at 14:07 UTC.'] },
    expected: 'The service recovered by 14:07 UTC, when the health check passed after errors fell to zero.' }),
  natural({ id: 'support-triage', name: 'Support triage', category: 'Composed agents',
    level: 'Advanced', description: 'Classify a ticket, then assemble a checked route and priority.',
    concepts: ['child calls', 'classification', 'typed composition'], root: 'support/triage.nl',
    source: `---
args:
  ticket: Text
returns: Triage
types:
  Triage: '{ team: Text, priority: Text }'
---
First call classify on args/ticket to choose a team. Then call prioritize on the same ticket. Write the two results into return.`,
    files: {
      'support/triage/classify.nl': `---
args:
  ticket: Text
returns: Text
---
Choose "billing" for payment problems and "technical" for product failures. Write the team to return.
`,
      'support/triage/prioritize.nl': `---
args:
  ticket: Text
returns: Text
---
Write "urgent" for a current outage; otherwise write "normal" to return.
`,
    },
    inputs: { ticket: 'The app is down for all users.' },
    expected: { team: 'technical', priority: 'urgent' } }),
  natural({ id: 'incident-brief', name: 'Incident brief', category: 'Composed agents',
    level: 'Advanced', description: 'Extract severity and assemble an incident report.',
    concepts: ['child calls', 'structured report'], root: 'incident/brief.nl',
    source: `---
args:
  log: Text
returns: Brief
types:
  Brief: '{ severity: Text, headline: Text }'
---
Call severity on args/log. Write the severity and a short factual headline to return.`,
    files: { 'incident/brief/severity.nl': `---
args:
  log: Text
returns: Text
---
Return "critical" when the log says the service is unavailable to all users; otherwise return "minor".
` },
    inputs: { log: 'Checkout was unavailable to all users for 12 minutes.' },
    expected: { severity: 'critical', headline: 'Checkout was unavailable to all users for 12 minutes.' } }),
  natural({ id: 'release-review', name: 'Release review', category: 'Composed agents',
    level: 'Advanced', description: 'Combine an exact test gate with a semantic release note.',
    concepts: ['crisp child', 'natural child', 'decision'], root: 'release/review.nl',
    source: `---
args:
  notes: Text
  tests_passed: Bool
returns: ReleaseReview
types:
  ReleaseReview: '{ ship: Bool, summary: Text }'
---
Call gate with tests_passed. Call summarize with notes. Write both results to return without changing the gate decision.`,
    files: {
      'release/review/gate.ts': `/*---
args:
  tests_passed: Bool
returns: Bool
engine: typescript-host
---*/
return args.tests_passed;
`,
      'release/review/summarize.nl': `---
args:
  notes: Text
returns: Text
---
Write a concise one-sentence summary of args/notes to return.
`,
    },
    inputs: { notes: 'The search page is faster and keyboard navigation is improved.', tests_passed: true },
    expected: { ship: true, summary: 'Search is faster and keyboard navigation is improved.' } }),
  natural({ id: 'survey-themes', name: 'Survey themes', category: 'Composed agents',
    level: 'Advanced', description: 'Classify each comment, then summarize the dominant theme.',
    concepts: ['Map', 'aggregation', 'child calls'], root: 'survey/themes.nl',
    source: `---
args:
  comments: Text[]
returns: ThemeReport
types:
  ThemeReport: '{ labels: Text[], summary: Text }'
---
For each comment in args/comments call label. Write the ordered labels and one sentence about the most common issue to return.`,
    files: { 'survey/themes/label.nl': `---
args:
  comment: Text
returns: Text
---
Use "speed" for performance complaints, "usability" for interface complaints, or "other". Write the label to return.
` },
    inputs: { comments: ['The page is slow.', 'Search takes forever.', 'The buttons are confusing.'] },
    expected: { labels: ['speed', 'speed', 'usability'], summary: 'Speed is the most common issue.' } }),
  natural({ id: 'migration-plan', name: 'Migration plan', category: 'Composed agents',
    level: 'Advanced', description: 'Produce ordered steps from a dependency list.',
    concepts: ['planning', 'dependencies', 'exact helper'], root: 'migration/plan.nl',
    source: `---
args:
  tasks: Task[]
returns: Text[]
types:
  Task: '{ id: Text, needs: Text[] }'
---
Call order_tasks with args/tasks. Write the returned order to return. If dependencies are impossible, report an error rather than guessing.`,
    files: { 'migration/plan/order_tasks.ts': `/*---
args:
  tasks: Task[]
returns: Text[]
engine: typescript-host
---*/
const done = [], pending = new Map(args.tasks.map(task => [task.id, task.needs]));
while (pending.size) {
  const ready = [...pending].filter(([, needs]) => needs.every(id => done.includes(id))).map(([id]) => id).sort();
  if (!ready.length) throw new Error('dependency cycle');
  for (const id of ready) { done.push(id); pending.delete(id); }
}
return done;
` },
    inputs: { tasks: [{ id: 'cutover', needs: ['backup', 'verify'] },
      { id: 'verify', needs: ['backup'] }, { id: 'backup', needs: [] }] },
    expected: ['backup', 'verify', 'cutover'] }),
];
