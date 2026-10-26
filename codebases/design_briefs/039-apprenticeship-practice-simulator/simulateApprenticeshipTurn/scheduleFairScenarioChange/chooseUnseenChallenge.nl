---
args: { family: string, objective: string, seed: string, seenIds: ID[] }
returns: ChallengeCandidate
---
Select from the scenario family's approved challenge catalog using the supplied seed and objective. Exclude any candidate whose ID is already in seenIds. Preserve the catalog's learning objective while choosing a different surface detail; do not change required procedure steps or difficulty without an explicit course revision.
