---
description: Join one location to editorial target decisions and source-family support.
args:
  state: ArbitrationFrontier
returns: ArbitrationFrontier
---
1. Select the first pending event ID and gather its actual WrittenEvent readings. Gather original
   editorial claims whose stated candidate scope can concern this event, including unauthorized claims.
2. Call resolveTargets(eventId, readings, claims, protocol) with those actual objects. Its scoped
   decisions distinguish applicability from authority and preserve source attachment ambiguity.
3. Call compareSupport(readings, claims, targets, families, protocol). Consume its actual ReadingBundle;
   do not rank copy-family members or apply a preferred correction before the child comparison.
4. Validate original event identity in each option and applied claim membership. A multi-target
   exclusive claim's alternatives must retain a consistent ClaimAssignment for later hypothesis joins.
5. Append the actual bundle, remove only this pending event ID and return the shorter frontier. Keep
   every previous bundle and unresolved source issue. A bundle with no established option stays a
   concrete unresolved location, not a zero-duration or default-pitch reading.
