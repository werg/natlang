---
args: {"promises": "PromiseEpisode[]", "slices": "OwnershipSlice[]", "docket": "DepartureDocket", "prior": "HandoverPortfolio | null"}
returns: "IntroductionIntent[]"
---
Find external commitments where the customer expects a named person. Draft a minimal introduction using the actual accepted successor and agreed contact scope; obtain wording review through the stated relationship owner. Preserve prior dispatch/receipt identities. An unreviewed draft stays pending; a timeout stays unknown and demands receipt reconciliation. Only an actual matched receipt records delivery. Do not expose private departure reasons or internal notes.
