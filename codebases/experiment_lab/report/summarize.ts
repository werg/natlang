import type { Case, Plan, Candidate, Trial, Metric, Analysis, Report } from "../types.js";
export default function summarize(candidates: Candidate[], trials: Trial[]): Metric[] {
return candidates.map(candidate => {
  const runs = trials.filter(t => t.candidate === candidate.id);
  const byCase = new Map();
  for (const trial of runs) {
    const list = byCase.get(trial.case_id) || [];
    list.push(trial);
    byCase.set(trial.case_id, list);
  }
  let repeats_compared = 0, repeats_agree = 0;
  for (const list of byCase.values()) {
    const completed = list.filter(t => t.status === "done" && t.value_digest);
    for (let i = 1; i < completed.length; i++) {
      repeats_compared++;
      if (completed[i].value_digest === completed[0].value_digest) repeats_agree++;
    }
  }
  return {
    candidate: candidate.id, planned: runs.length,
    done: runs.filter(t => t.status === "done").length,
    failed: runs.filter(t => t.status === "quiesced" || t.status === "exception").length,
    missing: runs.filter(t => t.status === "missing").length,
    provenance_ok: runs.filter(t => t.status === "done" && t.provenance).length,
    reviewed_pass: runs.filter(t => t.quality === "pass").length,
    reviewed_fail: runs.filter(t => t.quality === "fail").length,
    review_pending: runs.filter(t => t.quality === "pending").length,
    repeats_compared, repeats_agree
  };
});
}
