---
description: Determine whether a proposed collection use is covered by every relevant right, channel, audience and duration permission.
args: { dossier: RightsDossier; change: RightsChange; steward: RightsSteward }
returns: RightsProposal
---
Treat object ownership, reproduction, copyright, interview consent, attribution and online rights as separate permission records. Do not infer reproduction permission from object ownership.

Bind permissions = `extractUsePermissions(dossier, change)` and preserve each item's right type, channel, audience, attribution and expiry. Bind proposedUses = `interpretProposedUse(dossier, change)` to bind purpose, transformation, channel, audience and duration.

Bind compatibility = `intersectUseRights(dossier, change, permissions, proposedUses)`. For a derivative combining items, require every contributing item's permission to cover the complete use. A print license does not authorize online publication. Route missing rights to the relevant rightsholder.

Bind clearances = `createClearanceRequests(dossier, compatibility)` only for unresolved rights; unanswered requests remain pending and are never treated as grants.

Bind register = `maintainRightsRegister(dossier, change, compatibility, clearances)`. Schedule expiry review and removal for time-limited permissions. When attribution changes, update every affected print, web and promotional derivative with traceable versions.

Return `RightsProposal` for `steward.id` with bound permissions, compatibility, clearances, register, derivative states, expiry actions and a resumable use register. A proposed use is not cleared or published until the exact required rights are recorded.
