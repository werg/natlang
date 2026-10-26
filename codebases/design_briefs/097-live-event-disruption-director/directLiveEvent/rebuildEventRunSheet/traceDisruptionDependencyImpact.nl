---
description: Trace disruption impact through segment prerequisites, promises, resources and switch deadlines.
args:{event:EventProductionState;commitments:PromiseLedgerItem[];options:string[];checkpoint:LiveCheckpoint}
returns:{affectedSegmentIds:string[];downstreamSegmentIds:string[];preservedCommitmentIds:string[];atRiskCommitmentIds:string[];resourceIds:string[];unresolvedDependencyIds:string[];optionSwitchDeadlines:Record<string,number>}
---
Start from the exact affected segment IDs in confirmed disruption reports; follow each segment's prerequisites and resource dependencies forward through the run sheet. Bind each affected segment to its commitment IDs and responsible roles. Preserve commitments even when no option currently serves them; mark those outcomes at risk rather than deleting them. For each alternative, compute the last feasible switch from supplied setup, travel, booking and venue-close constraints; leave unknown duration or approval dependencies unresolved instead of inventing slack. Return the transitive impact set, preserved and at-risk promise IDs, consumed resources, unresolved edges and option deadlines.
