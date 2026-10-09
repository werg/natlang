---
description: The incident investigator for one event. Judge its significance, cluster it into incidents, generate and test hypotheses over the log index, decide whether to escalate, and summarize.
args:
  item: LogEvent
  observation?: Observation
  incidents: Incident[]
  settings: LogSettings
  files?: Folder
returns: Decision
---
Investigate item, one event of the log stream, against incidents, the incidents open before it, with the stages in your
folder. The index service is in the stages' reach. Return a Decision; the host commits it and carries out its effects.
Nothing here is sent or stored.

A gap (item.kind "gap"): note = gap(item, incidents). Return { significance: "ignore", incident: null, folded: [],
escalation: ignore, effects: [], gap: note }, where escalation is { action: "ignore", severity: "", claim: "",
uncertainty: "", hypothesis_id: "", cited: [] }.

A log line, in these steps:

1. Significance. s = significance(item, observation, settings). When s is "ignore" return a Decision with that
   significance, incident null, folded [], escalation ignore, effects [], gap null and status "observing".
2. Open incidents: those of incidents whose closes_at is at least item.occurred_at.
3. Clustering. a = cluster(item, the open incidents). Build the incident record `inc`:
   - "open": { id: item.id, opened_at: item.occurred_at, last_seen: item.occurred_at, closes_at: item.occurred_at +
     settings.quiet_ms, services: [item.service], codes: [item.code], members: [item.id], count: 1, severity: "",
     hypotheses: [], alert_key: null, summary: "" };
   - "join": the incident a.incident_id with last_seen the larger of its last_seen and item.occurred_at, closes_at the
     larger of its closes_at and item.occurred_at + settings.quiet_ms, item.service and item.code added to services and
     codes, item.id appended to members (keep the newest settings.max_members), count + 1;
   - folding: for each ID in a.fold, take that incident's members (added to inc.members, keeping the newest
     settings.max_members), count (added to inc.count), services, codes and hypotheses (added without repeats),
     its opened_at if smaller, and its alert_key if inc has none. `folded` is a.fold.
4. Gate. When inc.count is below settings.threshold and s is not "urgent", return the Decision with incident inc,
   escalation ignore and status "observing".
5. Runbook. note = runbook(item, files) when files is given, otherwise "".
6. Hypotheses. When inc.hypotheses is empty, or item.service or item.code was new to the incident in step 3,
   inc.hypotheses = hypothesize(inc, item, note).
7. Evidence. qs = queries(inc, inc.hypotheses, item, settings); found = gather(qs). For each hypothesis, all at once:
   weigh(hypothesis, the found entries with its ID). These are `supports`.
8. Escalation. e = escalate(inc, inc.hypotheses, supports, item, observation, settings). inc.severity = e.severity when
   it is not "".
9. Summary. When e.action is "escalate", inc.summary = summarize(inc, inc.hypotheses, supports, e).
10. Effects. When e.action is "escalate": inc.alert_key = inc.id, and effects = [{ kind: "alert", key: inc.id, service:
    item.service, code: item.code, claim: e.claim, evidence_ids: e.cited }]. Otherwise effects = [].
11. Status. status = "alerted" when e.action is "escalate", otherwise "investigating".
12. Return { significance: s, incident: inc, folded, escalation: e, effects, gap: null, status }.
