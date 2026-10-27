// The learned optimizer's own tuning score (ts-host/src/tune/reward.ts): a crisp verifier over the exact reference
// measurements that `python -m natlang_neuralese.tune.learned meta-examples` records with each example. Each stage is
// graded by the global target -- support-loss improvement of the adaptation -- not only by an exact label.

import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

// The support loss the services measured for a bundle of this context (apply.handle records each one under its id);
// null for a bundle the services never produced.
const measured = (out, id) => {
  if (typeof id !== 'string' || !/^[A-Za-z0-9._:-]+$/.test(id)) return null;
  const path = join(out, 'bundles', `${id}.json`);
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')).supportLoss : null;
};

const gain = (reference, operator) => operator === 'keep' ? 0 : Math.max(0, reference.incumbentLoss - reference.losses[operator]);

const scores = {
  // A whole episode: the share of the counterfactual episode's support-loss improvement that the returned bundle
  // achieves, by the services' own measurement (beating it scores 1). Returning no bundle keeps the incoming loss.
  'adapt.nl': (value, reference) => {
    const id = value?.current?.id ?? null;
    const loss = id === null ? reference.incomingLoss : measured(reference.out, id);
    if (loss === null) return 0;
    const best = reference.incomingLoss - reference.bestLoss, achieved = reference.incomingLoss - loss;
    if (best <= 0) return achieved >= 0 ? 1 : 0;
    return Math.max(0, Math.min(1, achieved / best));
  },
  // The share of the best available improvement the chosen operator achieves; "keep" is right only when nothing helps.
  'adapt/chooseOperator.nl': (value, reference) => {
    const best = Math.max(0, ...Object.keys(reference.losses).map(operator => gain(reference, operator)));
    if (best === 0) return value === 'keep' ? 1 : 0;
    return value in reference.losses || value === 'keep' ? gain(reference, value) / best : 0;
  },
  // Stopping when improvement remains loses that share of the episode's gain; continuing without any is wasted work.
  'adapt/shouldStop.nl': (value, reference) => {
    if (value === reference.stop) return 1;
    return value === true ? Math.max(0, 1 - reference.remainingShare) : 0;
  },
  // Per binding, the share of the best single-operation improvement the chosen operation achieves (a binding outside
  // focus is right only when kept); the plan's score is the mean over the program's bindings. The reason is free.
  'adapt/chooseRevision.nl': (value, reference, example) => {
    const chosen = value?.operations ?? {}, bindings = Object.keys(example.expected.operations);
    const binding = name => {
      const operation = chosen[name] ?? 'keep', losses = reference.losses[name];
      if (!losses) return operation === 'keep' ? 1 : 0;
      const gainOf = op => op === 'keep' || !(op in losses) ? 0 : Math.max(0, reference.incomingLoss - losses[op]);
      const best = Math.max(0, ...Object.keys(losses).map(gainOf));
      return best === 0 ? (operation === 'keep' ? 1 : 0) : gainOf(operation) / best;
    };
    return bindings.length ? bindings.reduce((sum, name) => sum + binding(name), 0) / bindings.length : 1;
  },
  // F1 of the chosen bindings against the hindsight focus.
  'adapt/chooseFocus.nl': (value, reference) => {
    const chosen = new Set(Array.isArray(value) ? value : []), label = new Set(reference.focus);
    if (!chosen.size && !label.size) return 1;
    const hits = [...chosen].filter(item => label.has(item)).length;
    return hits ? 2 * hits / (chosen.size + label.size) : 0;
  },
};

export default function reward(value, example) {
  const score = scores[example.function];
  if (!score || !example.reference) return JSON.stringify(value) === JSON.stringify(example.expected);
  return score(value, example.reference, example);
}
