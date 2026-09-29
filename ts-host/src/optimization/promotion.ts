import type { PreparedSuite } from '../evaluation/types.js';
import { candidateArtifact } from '../evaluation/runner.js';
import { artifactDigest } from '../adaptation/schema.js';
import type { SearchState } from './types.js';
export function promote(prepared: PreparedSuite, state: SearchState, finalized = state.selected) {
  const selected = state.population.find(candidate => candidate.id === state.incumbent) ?? state.baseline;
  const artifact = candidateArtifact(prepared, selected.value, state.runId);
  artifact.provenance = { ...artifact.provenance, promotion: finalized ? 'selected' : 'incumbent', strategy: state.strategy, engine: state.engine, seed: state.seed,
    evidence: { baselineQuality: state.baseline.validation.quality, selectedQuality: selected.validation.quality,
      gatesPassed: selected.validation.gatesPassed, validationDigest: selected.validation.digest,
      lockedTestDigest: state.lockedTest?.digest ?? null, ledger: state.ledger, stopReason: state.stopReason ?? null } };
  artifact.digest = artifactDigest(artifact); return artifact;
}
