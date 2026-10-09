import type { Untrusted } from '@natlang/node';

export type Recipe = { id: string, description: string };
/** What a job's result is: `unknown` when the command may or may not have run to completion. */
export type ResultStatus = "ok" | "failed" | "unknown";
/** Where a session stands. `running` and `cancel-requested` mean a job is active. */
export type SessionStatus = ResultStatus | "idle" | "running" | "cancel-requested" | "unsupported";
export type TerminalEvent = { kind: "request" | "complete" | "cancel" | "recover", id: string, request_id: string, job_id: string,
  text: string, status: ResultStatus | "", detail: string,
  /** True when a cancellation was requested for the job before it completed. The status and detail are its actual result. */
  cancel_requested: boolean };
/** A completion as the explanation reads it: the command's output is quoted data from outside the program. */
export type Completion = { request_id: string, job_id: string, status: ResultStatus, detail: Untrusted<string>, cancel_requested: boolean };
