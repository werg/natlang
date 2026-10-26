---
args: { identity: AssetIdentityEvidence }
returns: AssetResolution
---
Use only verified evidence that links an alias and location to one official asset ID. If several candidates remain or the alias is informal and unverified, return an unresolved identity and the exact conflict; do not choose the most familiar asset.
