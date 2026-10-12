/** Where a job goes. `queue-dgx` waits in the DGX memory ledger; `ask` hands an owner-visible decision to the owner. */
export type PlacementTarget = "dgx" | "pop" | "coreweave" | "queue-dgx" | "ask";

/** What one job needs, from its job file and measured peaks (scripts/coreweave/placement.py, job_needs). */
export type JobNeeds = {
  /** Job id and kind (smoke, eval, render, train, probe). */
  id: string,
  kind: string,
  /** Peak GPU memory in GB, measured on an earlier run when one exists, else declared. */
  gpu_memory_gb: number,
  /** Expected wall-clock hours. */
  hours: number,
  /** Talks to the teacher server on DGX (:8082), which is unreachable from CoreWeave. */
  needs_teacher: boolean,
  /** The machine that owns the job's lineage ("dgx" or "pop"). */
  lineage_owner: "dgx" | "pop",
  /** Moves the active training lineage of a line to another machine (owner-visible). */
  moves_active_lineage: boolean,
  /** Upper bound in USD for running it on CoreWeave for `hours`, or null while the GPU rate is unknown. */
  coreweave_cost_usd: number | null,
};

/** What the fleet looks like now, gathered by crisp code. */
export type FleetFacts = {
  /** DGX memory the job could be admitted into now (the memory ledger's headroom), GB. */
  dgx_headroom_gb: number,
  /** DGX ledger units waiting ahead of a job of this priority. */
  dgx_waiting_ahead: number,
  /** Credential, bucket and a passed GPU smoke: CoreWeave can take GPU jobs. */
  coreweave_ready: boolean,
  /** GPU memory of one CoreWeave serverless sandbox, GB (RTX PRO 6000: 96). */
  coreweave_gpu_memory_gb: number,
  /** Headroom kept free on that GPU for allocator fragmentation and peaks, GB. */
  coreweave_fit_margin_gb: number,
  /** Natlang sandboxes this machine has running, and the most it may run at once. */
  coreweave_active: number,
  coreweave_max_parallel: number,
  /** USD left of this week's CoreWeave budget, or null when no budget is set. */
  coreweave_budget_left_usd: number | null,
};

export type PlacementDecision = { where: PlacementTarget, reason: string };

/** One job as the bridge passes it: needs, fleet facts and the crisp implementation's decision. */
export type PlacementItem = { id: string, job: JobNeeds, fleet: FleetFacts, crisp: PlacementDecision };
