// The learned optimizer's own tuning score (ts-host/src/tune/reward.ts): a crisp verifier over the exact reference
// measurements that `python -m natlang_neuralese.tune.learned meta-examples` records with each example. Each stage is
// graded by the global target -- support-loss improvement of the adaptation -- not only by an exact label.

const gain = (reference, operator) => operator === 'keep' ? 0 : Math.max(0, reference.incumbentLoss - reference.losses[operator]);

const scores = {
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
  return score(value, example.reference);
}
