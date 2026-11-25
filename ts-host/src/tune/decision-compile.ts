/**
 * Single-pass calibrated `readout: decision` compilation & runtime settlement (`plans/NATLANG_COMPILER.md` §6.1, W2.1).
 *
 * Replaces 4x reasoning + 4x option-scoring loops on tuned finite-union decisions with
 * 1 prefill + calibrated temperature-scaled option scoring (`POST /v1/natlang/score`)
 * and margin/risk-gated fallback to full CoT or `escalate`.
 */

export const COMPILED_DECISION_SCHEMA = 'natlang.decision-compiled/1' as const;

export interface CompiledDecisionArtifact {
  schema: typeof COMPILED_DECISION_SCHEMA;
  siteId: string;
  labels: readonly string[];
  temperature: number;
  fastThreshold: number;
  riskThresholds?: Readonly<Record<string, number>>;
  adapterReleaseKey?: string | null;
  devMetrics?: {
    nllBefore: number;
    nllAfter: number;
    eceBefore: number;
    eceAfter: number;
    sampleCount: number;
  };
}

export interface DecisionCalibrationSample {
  /** Raw option log-probabilities / logits ordered to match `labels`. */
  logits: readonly number[];
  /** Index into `labels` of the verified expected label. */
  targetIndex: number;
}

export interface DecisionSettlement {
  settled: boolean;
  label: string;
  confidence: number;
  probabilities: Record<string, number>;
  reason: 'fast-path' | 'below-threshold' | 'risk-floor';
}

/** Numerically stable softmax over `logits / temperature`. */
export function softmaxWithTemperature(logits: readonly number[], temperature = 1.0): number[] {
  if (!logits.length) throw new RangeError('logits must be nonempty');
  if (!Number.isFinite(temperature) || temperature <= 0) {
    throw new RangeError(`temperature must be finite and positive, got ${temperature}`);
  }
  const scaled = logits.map(x => x / temperature);
  let maxVal = -Infinity;
  for (const v of scaled) if (v > maxVal) maxVal = v;
  let sum = 0;
  const exps = scaled.map(v => {
    const e = Math.exp(v - maxVal);
    sum += e;
    return e;
  });
  return exps.map(e => e / sum);
}

/** Mean negative log-likelihood on a set of calibration samples under `temperature`. */
export function decisionNll(samples: readonly DecisionCalibrationSample[], temperature: number): number {
  if (!samples.length) throw new RangeError('samples must be nonempty');
  let total = 0;
  for (const sample of samples) {
    const probs = softmaxWithTemperature(sample.logits, temperature);
    const p = probs[sample.targetIndex];
    if (p === undefined) throw new RangeError(`targetIndex ${sample.targetIndex} out of range`);
    total += -Math.log(Math.max(p, 1e-15));
  }
  return total / samples.length;
}

/** Expected Calibration Error (ECE) over top-1 predicted confidences. */
export function computeDecisionEce(
  samples: readonly DecisionCalibrationSample[],
  temperature = 1.0,
  numBins = 10,
): number {
  if (!samples.length) return 0;
  const counts = new Array<number>(numBins).fill(0);
  const accSums = new Array<number>(numBins).fill(0);
  const confSums = new Array<number>(numBins).fill(0);

  for (const sample of samples) {
    const probs = softmaxWithTemperature(sample.logits, temperature);
    let bestIdx = 0;
    let bestProb = -Infinity;
    for (let i = 0; i < probs.length; i++) {
      const p = probs[i]!;
      if (p > bestProb) {
        bestProb = p;
        bestIdx = i;
      }
    }
    const bin = Math.min(numBins - 1, Math.max(0, Math.floor(bestProb * numBins)));
    counts[bin] = (counts[bin] ?? 0) + 1;
    accSums[bin] = (accSums[bin] ?? 0) + (bestIdx === sample.targetIndex ? 1 : 0);
    confSums[bin] = (confSums[bin] ?? 0) + bestProb;
  }

  let ece = 0;
  for (let b = 0; b < numBins; b++) {
    const c = counts[b] ?? 0;
    if (c === 0) continue;
    const acc = (accSums[b] ?? 0) / c;
    const conf = (confSums[b] ?? 0) / c;
    ece += (c / samples.length) * Math.abs(acc - conf);
  }
  return ece;
}

/**
 * Deterministic golden-section search in log-temperature space minimizing dev NLL.
 * Falls back to temperature 1.0 if scaling does not strictly improve NLL.
 */
export function calibrateTemperature(
  samples: readonly DecisionCalibrationSample[],
  options: { minTemp?: number; maxTemp?: number; iterations?: number } = {},
): { temperature: number; nllBefore: number; nllAfter: number; eceBefore: number; eceAfter: number } {
  const minTemp = options.minTemp ?? 0.1;
  const maxTemp = options.maxTemp ?? 10.0;
  const iterations = options.iterations ?? 48;

  const nllBefore = decisionNll(samples, 1.0);
  const eceBefore = computeDecisionEce(samples, 1.0);

  const phi = (Math.sqrt(5) - 1) / 2;
  let lo = Math.log(minTemp);
  let hi = Math.log(maxTemp);
  let c = hi - phi * (hi - lo);
  let d = lo + phi * (hi - lo);
  let fc = decisionNll(samples, Math.exp(c));
  let fd = decisionNll(samples, Math.exp(d));

  for (let i = 0; i < iterations; i++) {
    if (fc < fd) {
      hi = d;
      d = c;
      fd = fc;
      c = hi - phi * (hi - lo);
      fc = decisionNll(samples, Math.exp(c));
    } else {
      lo = c;
      c = d;
      fc = fd;
      d = lo + phi * (hi - lo);
      fd = decisionNll(samples, Math.exp(d));
    }
  }
  const bestTemp = Math.exp(0.5 * (lo + hi));
  const nllAfter = decisionNll(samples, bestTemp);
  if (nllAfter >= nllBefore) {
    return { temperature: 1.0, nllBefore, nllAfter: nllBefore, eceBefore, eceAfter: eceBefore };
  }
  const eceAfter = computeDecisionEce(samples, bestTemp);
  return { temperature: bestTemp, nllBefore, nllAfter, eceBefore, eceAfter };
}

/**
 * Score a compiled single-pass decision and determine whether it settles locally or
 * falls back to full CoT / teacher escalation.
 */
export function scoreCompiledDecision(
  artifact: CompiledDecisionArtifact,
  rawOptionLogits: readonly number[],
): DecisionSettlement {
  if (rawOptionLogits.length !== artifact.labels.length) {
    throw new RangeError(
      `rawOptionLogits length (${rawOptionLogits.length}) must match labels (${artifact.labels.length})`,
    );
  }
  const probs = softmaxWithTemperature(rawOptionLogits, artifact.temperature);
  const probMap: Record<string, number> = {};
  let bestIdx = 0;
  let bestProb = -Infinity;
  for (let i = 0; i < artifact.labels.length; i++) {
    const label = artifact.labels[i]!;
    const p = probs[i]!;
    probMap[label] = p;
    if (p > bestProb) {
      bestProb = p;
      bestIdx = i;
    }
  }
  const bestLabel = artifact.labels[bestIdx]!;

  if (artifact.riskThresholds) {
    for (let i = 0; i < artifact.labels.length; i++) {
      const label = artifact.labels[i]!;
      const floor = artifact.riskThresholds[label];
      if (floor !== undefined && probs[i]! >= floor && label !== bestLabel) {
        return {
          settled: false,
          label: bestLabel,
          confidence: bestProb,
          probabilities: probMap,
          reason: 'risk-floor',
        };
      }
    }
  }

  if (bestProb < artifact.fastThreshold) {
    return {
      settled: false,
      label: bestLabel,
      confidence: bestProb,
      probabilities: probMap,
      reason: 'below-threshold',
    };
  }
  return {
    settled: true,
    label: bestLabel,
    confidence: bestProb,
    probabilities: probMap,
    reason: 'fast-path',
  };
}
