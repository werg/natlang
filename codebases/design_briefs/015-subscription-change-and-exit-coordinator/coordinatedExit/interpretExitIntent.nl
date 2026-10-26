---
description: Resolve requested subscription scope and distinguish product change from account closure.
args:{scope:GoalContract;request:SubscriptionExitRequest;subscriptions:SubscriptionCase[]}
returns:ChangeIntent|null
---
Match the exact customer, target IDs and expected version. Preserve requested effective date, unrequested linked services, retained data scope and explicit exclusions. Ask only when ambiguity changes access, billing or retention. Do not create consent or infer account closure from an add-on request.
