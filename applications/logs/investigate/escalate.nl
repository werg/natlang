---
description: Escalation decision and its evidence rule. Decide whether the incident is alerted, from the weighed hypotheses.
args:
  incident: Incident
  hypotheses: Hypothesis[]
  supports: Support[]
  item: LogEvent
  observation: Observation
  settings: LogSettings
returns: Escalation
---
Decide the action for incident after item. Work out the evidence rule in eval, with these steps.

1. For each hypothesis whose kind is "cause" or "impact": `cited` = the distinct IDs in the evidence_ids of its
   supports whose stance is "supports", sorted; `against` = the distinct IDs in the evidence_ids of its supports whose
   stance is "contradicts".
2. The rule is met by a hypothesis when `cited` holds at least settings.threshold IDs and `against` holds fewer IDs
   than `cited`. Take the met hypothesis with the most cited IDs as the winner.
3. A benign hypothesis with stance "supports" whose evidence_ids include every ID of the winner's `cited` explains the
   events: there is no winner.
4. Action:
   - "escalate" when there is a winner, incident.alert_key is null, and observation.late is false;
   - "investigate" when there is no winner, when one exists but the incident already has an alert_key, or when
     observation.late is true.
5. severity: decide(severity, winner's claim, number of cited IDs, incident.services) when there is a winner, otherwise
   the incident's own severity.
6. claim: one sentence stating what is wrong, from the winner's claim and the incident's services (empty without a
   winner). uncertainty: one sentence naming contradicting or missing evidence, empty when there is none.
   cited: the winner's `cited` (empty without a winner). hypothesis_id: the winner's ID or "".

Return { action, severity, claim, uncertainty, hypothesis_id, cited }.
