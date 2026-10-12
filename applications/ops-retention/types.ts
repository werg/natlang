/** The tier of one file: KEEP and ACTIVE are never deleted; PRUNE-AUTO is deleted by --apply; PRUNE-ASK only when the owner approves its path. */
export type RetentionTier = "KEEP" | "ACTIVE" | "PRUNE-AUTO" | "PRUNE-ASK";

/** Day thresholds of training/storage-policy.json. */
export type RetentionThresholds = {
  /** A run that changed more recently than this is still running or just finished. */
  finished_after_days: number,
  /** A checkpoint of a stopped run older than this may go. */
  aborted_after_days: number,
  /** A partial write older than this is a leftover. */
  leftover_after_days: number,
};

/** What a file is, by its name and suffix. */
export type RetentionCategory = "artifact-registered" | "corpus-registered" | "sync-revision" | "cache" | "export" |
  "optimizer-state" | "teacher-dump" | "weights-snapshot" | "checkpoint" | "dataset" | "corpus-unregistered" |
  "data-other" | "logs-metrics" | "artifact-unregistered" | "other";

/** The facts of one large file, computed by crisp code (scripts/storage_retention.py, retention_facts). */
export type RetentionFacts = {
  category: RetentionCategory,
  /** Days since the file last changed. */
  age_days: number,
  /** Days since any file of its run last changed. */
  run_age_days: number,
  /** A corpus or artifact manifest pins this file. */
  registered: boolean,
  /** A run script, recipe, registry, plan or certificate names this file. */
  named_by_reference: boolean,
  /** A run script, recipe, registry, plan or certificate names the file's directory. */
  directory_named_by_reference: boolean,
  /** A log, receipt, certificate, qualification or labels file. */
  record_file: boolean,
  /** Under a path the storage policy keeps. */
  policy_keep_path: boolean,
  /** Why the file is in use now (open by a process, a process works in its run, a ledger claim), or null. */
  in_use: string | null,
  /** A partial write (*.pending, *.tmp, *.partial). */
  partial_write: boolean,
  /** Its run or directory is marked aborted, failed or stopped. */
  run_marked_stopped: boolean,
  /** The step number in a checkpoint's name, or null. */
  step: number | null,
  /** The highest step among the step checkpoints of the same directory, or null. */
  latest_step_in_directory: number | null,
  /** How many step checkpoints the same directory holds. */
  step_files_in_directory: number,
  /** Named as a best or final weights file. */
  best_or_final: boolean,
};

export type RetentionVerdict = { tier: RetentionTier, reason: string };

/** One file as the bridge passes it: its facts and the crisp implementation's verdict. */
export type RetentionItem = { id: string, path: string, logical: string, bytes: number, facts: RetentionFacts, crisp: RetentionVerdict };
