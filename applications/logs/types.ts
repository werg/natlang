import type { Untrusted } from '@natlang/node';

/** Which implementation runs a hot-path policy. Both answer the same interface; the setting selects. */
export type Policy = "crisp" | "natlang";

/** The investigator's settings, passed to every stage that uses them. Times are milliseconds of event time. */
export type LogSettings = {
  /** Judging each log line's significance, once per line: "crisp" reads level and code, "natlang" reads the line. */
  significance: Policy,
  /** How far back evidence for one event reaches. */
  window_ms: number,
  /** How many distinct source records support a cause before it may be escalated. */
  threshold: number,
  /** How long an incident stays open after its last event. */
  quiet_ms: number,
  /** How many member IDs an incident remembers (the newest). */
  max_members: number,
};

/** A log line, or a gap in the source (kind "gap": records are missing around occurred_at). */
export type LogEvent = { kind: "log" | "gap", id: string, cursor: number, occurred_at: number, arrived_at: number,
  service: string, code: string, level: string,
  /** Text from outside the program: shown to a model as quoted data, never as instructions. */
  message: Untrusted<string> };
/** What the index knows of an ingested line. count is the lines of its service and code in the window ending at it. */
export type Observation = { id: string, status: "new" | "duplicate", service: string, code: string, occurred_at: number,
  count: number, late: boolean };
export type Evidence = { id: string, service: string, code: string, occurred_at: number, level: string, message: Untrusted<string> };

// The exact index service ---------------------------------------------------------------------------------------

/** All fields given must match. contains is a case-insensitive substring of the message. from and to are inclusive event times. */
export type SearchQuery = { service?: string, code?: string, level?: string, contains?: string, from: number, to: number, limit: number };
/** total counts every match; evidence holds the first `limit`, by event time then ID. */
export type SearchResult = { total: number, evidence: Evidence[] };

// The investigation -----------------------------------------------------------------------------------------------

/** ignore: not worth tracking. watch: track it in an incident. urgent: investigate at once. */
export type Significance = "ignore" | "watch" | "urgent";
export type Severity = "low" | "medium" | "high" | "critical";
export type Kin = "same" | "different";

/** cause: something that made the events happen. impact: what the events did. benign: an ordinary explanation (a test, a health check). */
export type Hypothesis = { id: string, claim: string, kind: "cause" | "impact" | "benign" };

/** A group of events taken to be one thing going wrong. Its id is the ID of the event that opened it. */
export type Incident = {
  id: string, opened_at: number, last_seen: number,
  /** Event time after which the incident is closed if nothing else joins. */
  closes_at: number,
  services: string[], codes: string[],
  /** The newest member event IDs, at most max_members. */
  members: string[],
  /** How many events joined in all. */
  count: number,
  severity: Severity | "",
  hypotheses: Hypothesis[],
  /** The idempotency key of the alert sent for this incident, or null. */
  alert_key: string | null,
  summary: string,
};
export type ClosedIncident = { id: string, closed_at: number, count: number, severity: Severity | "", alert_key: string | null, summary: string };

/** join: the event belongs to incident_id. open: it starts incident_id (its own ID). fold: other open incidents that are the same thing and merge into incident_id. */
export type Attachment = { action: "join" | "open", incident_id: string, fold: string[], reason: string };

/** One search the hypothesis calls for, over the exact index. */
export type Query = { hypothesis_id: string, service?: string, code?: string, level?: string, contains?: string, from: number, to: number, limit: number };
export type Found = { hypothesis_id: string, query: Query, total: number, evidence: Evidence[] };
export type Support = { hypothesis_id: string, stance: "supports" | "contradicts" | "neutral", evidence_ids: string[], note: string };

/** ignore: nothing to look into yet. investigate: looked, not enough to alert. escalate: alert. */
export type Escalation = { action: "ignore" | "investigate" | "escalate", severity: Severity | "", claim: string,
  uncertainty: string, hypothesis_id: string, cited: string[] };

/** Effects are data: the host carries them out and records receipts. */
export type Effect = { kind: "alert", key: string, service: string, code: string, claim: string, evidence_ids: string[] };

/** Which open incidents a source gap touches. */
export type GapNote = { affected: string[], unknown: string };

/** What the investigation decided for one event; the host commits it. */
export type Decision = { significance: Significance, incident: Incident | null, folded: string[], escalation: Escalation,
  effects: Effect[], gap: GapNote | null };

export type Alert = { status: "local" | "sent" | "unknown" | "duplicate" | "insufficient", key: string, detail: string };
export type IncidentState = { cursor: number, observed: number, alerts: Alert[], unknowns: string[],
  status: "idle" | "observing" | "investigating" | "alerted" | "delivery-unknown" | "duplicate" | "gap",
  incidents: Incident[], closed: ClosedIncident[] };
