---
args: { prepared: PreparedReplicationDependency, goal: GoalContract, policy: PolicySelection }
returns: DecisionsReplicationDependency
---
1. Apply this domain algorithm to each relevant evidence unit: Identify data, software, configuration, and computational dependencies for each step. Detect unavailable versions and incompatible assumptions, then formulate approved alternatives without representing them as exact replication of the original environment.
2. Return field-level atoms for the actual ReplicationDependency record. Each atom gives a value, evidence references, a status of established, disputed, unknown, or inapplicable, and a plausible alternative.
3. Distinguish an attributed fact from a suggestion, inference, repeated copy, or silence. Respect effective date, source authority, and the exact scope.
4. Preserve unresolved conflicts and ask the smallest question that could change the disposition.
