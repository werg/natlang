export type CollectorRequestTelemetryStart = {
  attempt_id?: string;
  purpose?: string;
  role?: string;
  planner_attempt?: number | null;
  plan_status?: string | null;
  status?: string;
  chat_transport_starts?: number;
  chat_transport_retry_starts?: number;
  provider_sdk_turn_starts?: number;
};

export type CollectorRequestTelemetryRow = {
  provenance?: { reused_from?: unknown };
  request_telemetry?: {
    attempt_ids?: string[];
    starts?: CollectorRequestTelemetryStart[];
    authored_synthetic_root_actions?: number;
  };
};

export type CollectorRequestTelemetrySummary = {
  rows_with_telemetry: number;
  rows_without_telemetry: number;
  rows_reused_with_telemetry: number;
  rows_fresh_or_resumed_with_telemetry: number;
  collector_attempts: number;
  collector_retry_attempts: number;
  collector_sender_starts: number;
  fresh_or_resumed_sender_starts: number;
  reused_lineage_sender_starts: number;
  planner_starts: number;
  planner_retry_starts: number;
  action_starts: number;
  judge_starts: number;
  chat_transport_starts: number;
  chat_transport_retry_starts: number;
  provider_sdk_turn_starts: number;
  planner_chat_transport_starts: number;
  action_chat_transport_starts: number;
  judge_chat_transport_starts: number;
  planner_provider_sdk_turn_starts: number;
  action_provider_sdk_turn_starts: number;
  judge_provider_sdk_turn_starts: number;
  planned_action_turns: number;
  planner_fallback_turns: number;
  sender_completed: number;
  sender_failed: number;
  authored_synthetic_root_actions: number;
};

export class CollectorRequestTelemetryAccumulator {
  readonly totals: CollectorRequestTelemetrySummary = {
    rows_with_telemetry: 0, rows_without_telemetry: 0, rows_reused_with_telemetry: 0,
    rows_fresh_or_resumed_with_telemetry: 0, collector_attempts: 0, collector_retry_attempts: 0,
    collector_sender_starts: 0, fresh_or_resumed_sender_starts: 0, reused_lineage_sender_starts: 0,
    planner_starts: 0, planner_retry_starts: 0, action_starts: 0, judge_starts: 0,
    chat_transport_starts: 0, chat_transport_retry_starts: 0, provider_sdk_turn_starts: 0,
    planner_chat_transport_starts: 0, action_chat_transport_starts: 0, judge_chat_transport_starts: 0,
    planner_provider_sdk_turn_starts: 0, action_provider_sdk_turn_starts: 0, judge_provider_sdk_turn_starts: 0,
    planned_action_turns: 0, planner_fallback_turns: 0, sender_completed: 0, sender_failed: 0,
    authored_synthetic_root_actions: 0,
  };

  add(row: CollectorRequestTelemetryRow): void {
    const telemetry = row.request_telemetry;
    if (!telemetry) {
      this.totals.rows_without_telemetry++;
      return;
    }
    const reused = row.provenance?.reused_from !== undefined;
    this.totals.rows_with_telemetry++;
    if (reused) this.totals.rows_reused_with_telemetry++;
    else this.totals.rows_fresh_or_resumed_with_telemetry++;
    this.totals.authored_synthetic_root_actions += telemetry.authored_synthetic_root_actions ?? 0;
    const starts = telemetry.starts ?? [];
    const attempts = new Set([...(telemetry.attempt_ids ?? []), ...starts
      .map(start => start.attempt_id).filter((id): id is string => Boolean(id))]);
    this.totals.collector_attempts += attempts.size;
    this.totals.collector_retry_attempts += Math.max(0, attempts.size - 1);
    for (const start of starts) {
      this.totals.collector_sender_starts++;
      if (reused) this.totals.reused_lineage_sender_starts++;
      else this.totals.fresh_or_resumed_sender_starts++;
      this.totals.chat_transport_starts += start.chat_transport_starts ?? 0;
      this.totals.chat_transport_retry_starts += start.chat_transport_retry_starts ?? 0;
      this.totals.provider_sdk_turn_starts += start.provider_sdk_turn_starts ?? 0;
      const purpose = start.purpose === 'planner' ? 'planner' :
        start.purpose === 'judge' || start.role === 'judge' ? 'judge' : 'action';
      this.totals[`${purpose}_chat_transport_starts`] += start.chat_transport_starts ?? 0;
      this.totals[`${purpose}_provider_sdk_turn_starts`] += start.provider_sdk_turn_starts ?? 0;
      if (start.purpose === 'planner') {
        this.totals.planner_starts++;
        if ((start.planner_attempt ?? 0) > 1) this.totals.planner_retry_starts++;
      } else if (start.purpose === 'judge' || start.role === 'judge') this.totals.judge_starts++;
      else {
        this.totals.action_starts++;
        if (start.plan_status === 'planned') this.totals.planned_action_turns++;
        if (start.plan_status === 'fallback') this.totals.planner_fallback_turns++;
      }
      if (start.status === 'completed') this.totals.sender_completed++;
      if (start.status === 'failed') this.totals.sender_failed++;
    }
  }
}
