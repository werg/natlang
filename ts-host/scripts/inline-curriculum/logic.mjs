// Logic families: entailment with late exceptions, verifier-checked proofs, and abduction.
import { createHash } from 'node:crypto';
import { Random, capitalize, curriculumCase, evalCall, literal, nonceWords, returnCall } from './lib.mjs';

const PAGE = 8;

/** A paged fact store as a callable-folder TypeScript module. */
export function factStore(facts, description) {
  return `const FACTS: { id: string, text: string }[] = ${literal(facts)};
/** ${description} Page n of the facts in store order, ${PAGE} per page; pages are numbered from 1 to pages(). */
export function page(n: number): { id: string, text: string }[] {
  if (!Number.isInteger(n) || n < 1 || n > ${Math.ceil(facts.length / PAGE)}) throw new RangeError('page ' + n + ' does not exist; pages run from 1 to ${Math.ceil(facts.length / PAGE)}');
  return FACTS.slice((n - 1) * ${PAGE}, n * ${PAGE});
}
/** The number of pages of facts. */
export function pages(): number { return ${Math.ceil(facts.length / PAGE)}; }
`;
}
export const READ_ALL = `const all: { id: string, text: string }[] = [];
const count = facts.pages();
for (let p = 1; p <= count; p++) all.push(...facts.page(p));
for (const f of all) console.log(f.id + ': ' + f.text);`;

/** Place `late` facts on the last page and `early` on the first, filling with shuffled filler; returns ids. */
const PADDING = ['The store was last audited in spring.', 'Two entries were merged during the audit.',
  'The archive keeps a paper copy of every entry.', 'Entries are reviewed once a season.', 'One entry was withdrawn as a duplicate.',
  'The index was rebuilt after the move.'].map(text => ({ text }));
function layout(rng, early, filler, late) {
  // Pad so the late facts start two pages in, past anything a first read shows.
  const middle = [...rng.shuffle(filler), ...PADDING].slice(0, Math.max(filler.length, PAGE * 2 - early.length));
  return [...early, ...rng.shuffle(middle), ...rng.shuffle(late)].map((fact, i) => ({ ...fact, id: `F${i + 1}` }));
}
const idOf = (facts, text) => facts.find(fact => fact.text === text).id;

// Real categories whose local rule contradicts ordinary knowledge.
const COUNTER_COMMONSENSE = [
  { kind: 'penguin', plural: 'penguins', verb: 'fly' }, { kind: 'snail', plural: 'snails', verb: 'outrun a horse' },
  { kind: 'goldfish', plural: 'goldfish', verb: 'climb trees' }, { kind: 'tortoise', plural: 'tortoises', verb: 'swim across the lake in a minute' },
  { kind: 'cactus', plural: 'cacti', verb: 'grow underwater' }, { kind: 'ostrich', plural: 'ostriches', verb: 'hover' },
];

/** Entailment against a quantified rule with a late exception, a decoy, or a missing membership fact. */
export function entailmentException(seed, index) {
  const rng = new Random(seed, `entailment:${index}`);
  const used = new Set();
  const counter = index % 3 === 2;
  const [kindWord, verbWord, cond, place, ...names] = nonceWords(rng, 12, used);
  const world = counter ? rng.pick(COUNTER_COMMONSENSE) : { kind: kindWord, plural: `${kindWord}s`, verb: verbWord };
  const [target, decoy, other1, other2, other3] = names.map(capitalize);
  const general = `Every ${world.kind} in the ${capitalize(place)} reserve can ${world.verb}.`;
  const exceptionRule = `A ${world.kind} that is ${cond} cannot ${world.verb}.`;
  const member = `${target} is a ${world.kind} in the ${capitalize(place)} reserve.`;
  const decoyMember = `${decoy} is a ${world.kind} in the ${capitalize(place)} reserve.`;
  const [otherKind, otherVerb, otherCond] = nonceWords(rng, 3, used);
  const filler = [
    `${other1} is a ${otherKind} in the ${capitalize(place)} reserve.`, `Every ${otherKind} can ${otherVerb}.`,
    `${other2} is ${otherCond}.`, `A ${otherKind} that is ${otherCond} sleeps through winter.`,
    `${other3} arrived at the reserve last year.`, `The ${capitalize(place)} reserve has two feeding stations.`,
    `${decoy} was tagged by the rangers in spring.`, `${other1} shares an enclosure with ${other3}.`,
    `Rangers count the ${world.plural} every month.`, `${other2} is a ${otherKind} outside the reserve.`,
    `The eastern gate of the reserve is closed at night.`, `${target} was weighed in autumn.`,
  ].map(text => ({ text }));
  const variants = {
    applies: { late: [exceptionRule, `${target} is ${cond}.`], early: [general, member, decoyMember],
      verdict: 'contradicted', exception: `${target} is ${cond}.` },
    decoy: { late: [exceptionRule, `${decoy} is ${cond}.`], early: [general, member, decoyMember],
      verdict: 'entailed', exception: null },
    unknown: { late: [exceptionRule, `${target} was hatched near the ${capitalize(place)} reserve.`],
      early: [general, decoyMember], verdict: 'unknown', exception: null },
  };
  const shape = `${counter ? 'counter' : 'nonce'}${index}`;
  return Object.entries(variants).map(([variant, v]) => {
    const facts = layout(rng, v.early.map(text => ({ text })), filler, v.late.map(text => ({ text })));
    const expected = { verdict: v.verdict, exception: v.exception ? { rule: idOf(facts, exceptionRule), trigger: idOf(facts, v.exception) } : null };
    const lateMarker = v.late[1];
    return curriculumCase({ family: 'logic_entailment_exception', shape, variant, pairGroup: `entailment:${shape}`,
      slice: 'observation_followup', domain: 'logic', mode: 'followup', worldSemantics: 'open_world',
      evidence: { world: facts.map(f => `${f.id}: ${f.text}`), retrieved: [exceptionRule, lateMarker],
        background: counter ? [`Ordinary knowledge says a ${world.kind} cannot ${world.verb}; the local rule overrides it.`] : [] },
      assumptions: ['Facts absent from the store are unknown, not false.'],
      decisive: [{ marker: exceptionRule, source: 'eval', note: 'the exception rule is on the last page' },
        { marker: lateMarker, source: 'eval', note: 'the late fact decides whether the exception applies' }],
      plausibleActions: ['answer entailed from the general rule', 'answer contradicted by an exception', 'answer unknown'],
      minimumSequence: ['read every page of the fact store', 'match the exception against the entity', 'return the judgment'],
      reference: { root: [evalCall(READ_ALL), returnCall(expected)] },
      root: { name: 'judge_claim', args: { question: 'string' }, returns: 'Judgment',
        instructions: `Answer question using only the facts in the store (facts). Do not use outside knowledge about the kinds of creature it names.
A fact that is not in the store is unknown, not false.
verdict is "entailed" when the facts establish a yes, "contradicted" when they establish a no, and "unknown" otherwise.
When an exception rule stops the general rule from applying to the questioned creature, exception gives the id of that exception rule (rule) and the id of the fact about the creature that makes it apply (trigger); otherwise exception is null.` },
      files: { 'judge_claim/facts.ts': factStore(facts, 'The reserve fact store.'),
        'types.ts': 'export type Judgment = { verdict: "entailed" | "contradicted" | "unknown", exception: { rule: string, trigger: string } | null };\n' },
      inputs: { question: `Can ${target} ${world.verb}?` }, expected });
  });
}

/** Proof with a verifier: a guarded rule whose negated condition is defeated by a late fact, or a missing link. */
export function proofVerifier(seed, index) {
  const rng = new Random(seed, `proof:${index}`);
  const used = new Set();
  const [k0, k1, k2, k3, k2x, z, ...names] = nonceWords(rng, 10, used);
  const [entity, other1, other2] = names.map(capitalize);
  const cert = createHash('sha256').update(`${seed}:${index}:${entity}:${k3}`).digest('hex').slice(0, 10);
  const base = { member: `${entity} is a ${k0}.`, r1: `Every ${k0} is a ${k1}.`, r2: `Every ${k1} is a ${k2}.`,
    r2decoy: `Every ${k1} is a ${k2x}.`, guarded: `Every ${k2} that is not ${z} is a ${k3}.`, blocker: `${entity} is ${z}.` };
  const filler = [`${other1} is a ${k1}.`, `${other2} is ${z}.`, `Every ${k2x} is a ${k0}.`, `${other1} is ${z}.`,
    `${other2} is a ${k2x}.`, `Every ${z} thing is fragile.`, `${other1} was catalogued first.`,
    `${other2} is kept in the north hall.`, `Every ${k3} is catalogued twice.`, `${entity} was catalogued last.`].map(text => ({ text }));
  const variants = {
    proved: { early: [base.member, base.r1], late: [base.r2, base.guarded], status: 'proved', blocking: null },
    blocked: { early: [base.member, base.r1], late: [base.r2, base.guarded, base.blocker], status: 'unprovable', blocking: base.blocker },
    missing: { early: [base.member, base.r1], late: [base.r2decoy, base.guarded], status: 'unprovable', blocking: null },
  };
  const shape = `chain${index}`;
  return Object.entries(variants).map(([variant, v]) => {
    const facts = layout(rng, v.early.map(text => ({ text })), filler, v.late.map(text => ({ text })));
    const ids = Object.fromEntries(facts.map(f => [f.text, f.id]));
    const expected = { status: v.status, blocking: v.blocking ? ids[v.blocking] : null,
      certificate: v.status === 'proved' ? `cert-${cert}` : null };
    const verifier = `const FACTS: Record<string, string> = ${literal(Object.fromEntries(facts.map(f => [f.id, f.text])))};
type Step = { claim: string, rule?: string, from?: string };
const norm = (text: string) => text.trim().replace(/\\.$/, '').replace(/ is an /, ' is a ');
const fail = (problem: string) => ({ ok: false, certificate: null, problem });
/**
 * Check a proof, one step per claim, in order. A step restates a fact (from: its id, no rule); states
 * "X is not B", which holds when the theory does not say "X is B" (closed world); or applies rule (the id of
 * an "Every ..." rule) to from (a fact id or an earlier step, "s0", "s1", ...). A proof whose last claim is
 * the goal returns a certificate.
 */
export function verify(steps: Step[]): { ok: boolean, certificate: string | null, problem: string | null } {
  const claims: string[] = [];
  for (let i = 0; i < steps.length; i++) {
    const step = steps[i], claim = norm(step.claim);
    const negated = /^(\\w+) is not (\\w+)$/.exec(claim);
    if (negated) {
      for (const [id, text] of Object.entries(FACTS)) if (norm(text) === negated[1] + ' is ' + negated[2])
        return fail('step ' + i + ': ' + id + ' states "' + text + '", so "' + claim + '" does not hold');
      claims.push(claim); continue;
    }
    const premise = step.from === undefined ? undefined : /^s\\d+$/.test(step.from) ? claims[Number(step.from.slice(1))] : FACTS[step.from] && norm(FACTS[step.from]);
    if (premise === undefined) return fail('step ' + i + ': from must name a fact id or an earlier step ("s0", ...), not ' + JSON.stringify(step.from));
    if (!step.rule || step.rule === step.from) {
      if (premise !== claim) return fail('step ' + i + ': ' + step.from + ' says "' + premise + '", not "' + claim + '"');
      claims.push(claim); continue;
    }
    const rule = FACTS[step.rule];
    if (rule === undefined) return fail('step ' + i + ': unknown rule ' + step.rule);
    const who = /^(\\w+) is a (\\w+)$/.exec(premise);
    const plain = /^Every (\\w+) is an? (\\w+)\\.$/.exec(rule);
    const guarded = /^Every (\\w+) that is not (\\w+) is an? (\\w+)\\.$/.exec(rule);
    const parts = plain ? [plain[1], null, plain[2]] : guarded ? [guarded[1], guarded[2], guarded[3]] : null;
    if (!parts) return fail('step ' + i + ': ' + step.rule + ' ("' + rule + '") is not an "Every ..." rule');
    if (!who || who[2] !== parts[0]) return fail('step ' + i + ': ' + step.rule + ' applies to a ' + parts[0] + ', and "' + premise + '" is not about one');
    if (parts[1]) for (const [id, text] of Object.entries(FACTS)) if (norm(text) === who[1] + ' is ' + parts[1])
      return fail('step ' + i + ': ' + id + ' states "' + text + '", so ' + step.rule + ' does not apply to ' + who[1]);
    const derived = who[1] + ' is a ' + parts[2];
    if (claim !== derived) return fail('step ' + i + ': ' + step.rule + ' gives "' + derived + '", not "' + claim + '"');
    claims.push(derived);
  }
  const goal = claims[claims.length - 1] === ${JSON.stringify(`${entity} is a ${k3}`)};
  return { ok: goal, certificate: goal ? ${JSON.stringify(`cert-${cert}`)} : null, problem: goal ? null : 'the last claim is not the goal' };
}
`;
    const proof = [{ claim: `${entity} is a ${k1}`, rule: ids[base.r1], from: ids[base.member] },
      { claim: `${entity} is a ${k2}`, rule: ids[base.r2] ?? ids[base.r2decoy], from: 's0' },
      { claim: `${entity} is a ${k3}`, rule: ids[base.guarded], from: 's1' }];
    const reference = variant === 'missing' ? [evalCall(READ_ALL), returnCall(expected)] :
      [evalCall(READ_ALL), evalCall(`const check = proof.verify(${JSON.stringify(proof)});\ncheck`), returnCall(expected)];
    const verifierOutput = variant === 'blocked' ? `${ids[base.blocker]} states` : `cert-${cert}`;
    return curriculumCase({ family: 'logic_proof_verifier', shape, variant, pairGroup: `proof:${shape}`,
      slice: 'observation_followup', domain: 'logic', mode: 'followup', worldSemantics: 'closed_world',
      evidence: { world: facts.map(f => `${f.id}: ${f.text}`), retrieved: [base.guarded], background: [] },
      assumptions: ['"not B" holds when the theory does not state it (closed world).'],
      decisive: [{ marker: base.guarded, source: 'eval', note: 'the guarded rule is on the last page' },
        ...(variant === 'missing' ? [{ marker: base.r2decoy, source: 'eval', note: 'the only rule out of the middle kind leads elsewhere' }] :
          [{ marker: verifierOutput, source: 'eval', note: 'the verifier accepts or names the defeating fact' }])],
      plausibleActions: ['return proved with a certificate', 'report the proof unprovable', 'repair the proof with another rule'],
      minimumSequence: ['read the theory', 'submit a proof to verify', 'interpret the verdict'],
      reference: { root: reference },
      root: { name: 'prove_goal', args: { goal: 'string' }, returns: 'ProofResult',
        instructions: `Decide whether the theory in facts proves goal.
The theory is closed: "X is not B" holds exactly when the theory does not state "X is B".
Check a candidate proof with proof.verify before you report "proved"; certificate is the certificate it returns.
When the goal cannot be proved, status is "unprovable", certificate is null, and blocking is the id of the fact that defeats a rule the proof needs, or null when a needed rule is simply absent.` },
      files: { 'prove_goal/facts.ts': factStore(facts, 'The catalogue theory.'), 'prove_goal/proof.ts': verifier,
        'types.ts': 'export type ProofResult = { status: "proved" | "unprovable", blocking: string | null, certificate: string | null };\n' },
      inputs: { goal: `${entity} is a ${k3}.` }, expected });
  });
}

const ABDUCTION = [
  { system: 'the packing-line conveyor', symptom: 'The conveyor makes a high whine and cartons arrive late. The whine began after the drive belt was replaced.',
    hypotheses: { belt: 'The new drive belt is too tight: it whines while it is driven, and the noise disappears when the belt is taken off.',
      bearing: 'A roller bearing is worn: it whines whenever the roller turns, belt or not, and it warms up quickly.' },
    test: { name: 'run_without_belt', description: 'Take the belt off and spin the rollers by hand, then listen.' },
    outcomes: { belt: 'With the belt off, the rollers turn silently.', bearing: 'With the belt off, the whine continues as the rollers turn.',
      unclear: 'With the belt off, the rollers could not be turned fast enough to tell whether the whine continues.' } },
  { system: 'the greenhouse tomatoes', symptom: 'Lower tomato leaves are yellowing. The yellowing started after the watering timer was changed.',
    hypotheses: { overwatering: 'Overwatering: the soil stays soggy an hour after watering and the roots are brown and soft.',
      nitrogen: 'Nitrogen deficiency: the soil drains normally, the roots are white and firm, and the yellowing moves upward.' },
    test: { name: 'inspect_roots', description: 'Lift one plant an hour after watering and examine its soil and roots.' },
    outcomes: { overwatering: 'The soil was still soggy and the roots were brown and mushy.', nitrogen: 'The soil had drained and the roots were white and firm.',
      unclear: 'The plant broke while it was lifted, so the soil and roots could not be examined.' } },
  { system: 'the office network', symptom: 'Video calls stutter every afternoon. It started after a new wireless access point was installed.',
    hypotheses: { interference: 'Radio interference on the new access point: wired computers are unaffected.',
      uplink: 'The internet uplink is saturated: wired and wireless computers stutter alike.' },
    test: { name: 'wired_call', description: 'Make an afternoon call from a computer plugged into the wired network.' },
    outcomes: { interference: 'The wired call was smooth all afternoon.', uplink: 'The wired call stuttered just like the wireless ones.',
      unclear: 'No wired computer was available this afternoon, so the call was not made.' } },
  { system: 'the bakery sourdough', symptom: 'Loaves come out dense. It began when the bakery moved to a cooler room.',
    hypotheses: { temperature: 'A cold proof: the dough rises normally once it proofs in a warm cabinet.',
      starter: 'A weak starter: the starter does not double within eight hours, even when kept warm.' },
    test: { name: 'warm_proof', description: 'Proof one batch in the warm cabinet and time the starter.' },
    outcomes: { temperature: 'In the warm cabinet the dough rose normally and the starter doubled in five hours.',
      starter: 'Even in the warm cabinet, the starter had not doubled after eight hours and the loaf was dense.',
      unclear: 'The warm cabinet failed overnight, so the batch proofed in the cool room again.' } },
];

/** Abduction: two causes explain the symptom; a discriminating test confirms one, reverses the prior, or is inconclusive. */
export function abductionTest(seed, index) {
  const scenario = ABDUCTION[index % ABDUCTION.length];
  const round = Math.floor(index / ABDUCTION.length);
  const [prior, alternative] = Object.keys(scenario.hypotheses);
  const shape = `${scenario.test.name}${round}`;
  const variants = { confirms: prior, reverses: alternative, inconclusive: null };
  return Object.entries(variants).map(([variant, cause]) => {
    const outcome = scenario.outcomes[cause ?? 'unclear'];
    const module = `const HYPOTHESES = ${literal(Object.entries(scenario.hypotheses).map(([name, explains]) => ({ name, explains })))};
/** The candidate causes and what each one predicts. */
export function hypotheses(): { name: string, explains: string }[] { return HYPOTHESES; }
/** The discriminating tests that can be run. */
export function tests(): { name: string, description: string }[] { return [${literal(scenario.test)}]; }
/** Run a test by name and report what was observed. */
export function run_test(name: string): string {
  if (name !== ${JSON.stringify(scenario.test.name)}) throw new Error('unknown test ' + name);
  return ${JSON.stringify(outcome)};
}
`;
    return curriculumCase({ family: 'logic_abduction_test', shape, variant, pairGroup: `abduction:${shape}`,
      slice: 'observation_followup', domain: 'logic', mode: 'followup', worldSemantics: 'defeasible',
      evidence: { world: Object.values(scenario.hypotheses), retrieved: [outcome],
        background: ['A test result that matches only one hypothesis\'s prediction discriminates between them.'] },
      assumptions: ['The symptom\'s timing is suggestive but not decisive.'],
      decisive: [{ marker: outcome, source: 'eval', note: 'the test result' }],
      plausibleActions: [`blame ${prior} from the timing`, `blame ${alternative}`, 'report the cause as undetermined'],
      minimumSequence: ['read the hypotheses', 'run the discriminating test', 'rank against the result'],
      reference: { root: [evalCall('JSON.stringify({ hypotheses: diagnosis.hypotheses(), tests: diagnosis.tests() })'),
        evalCall(`diagnosis.run_test(${JSON.stringify(scenario.test.name)})`), returnCall({ cause })] },
      root: { name: 'diagnose', args: { report: 'string' }, returns: 'Diagnosis',
        instructions: `Find the cause of the problem described in report for ${scenario.system}.
The candidate causes and the available tests are in diagnosis. The report's timing only suggests a cause; base the answer on test evidence.
cause is the name of the hypothesis the evidence supports, or null when the evidence cannot tell them apart.` },
      files: { 'diagnose/diagnosis.ts': module, 'types.ts': 'export type Diagnosis = { cause: string | null };\n' },
      inputs: { report: scenario.symptom }, expected: { cause } });
  });
}
