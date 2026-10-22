# Completed continuation scheduling authority

Policy remains the production recipe's declared RL scheduling policy. A completed
continuation must retain the earned checkpoint, optimizer/RNG and exposure while
binding its scheduling controls to the final sealed plan. The preparation plan
cannot include its own completed-adaptation proof, so its identity differs from
the final plan.

Crisp configuration mechanism:

1. Authenticate the completed adaptation and its exact future configuration.
2. Validate and rebind every scheduling control, including independent RL, from
   the authenticated old/preparation plan to the actual final plan.
3. If the exact previously configured candidate differs only by the unrebound RL
   control, require unchanged earned cursor/state and no active operation. Record
   immutable before/after configuration evidence under the existing leases and
   apply that configuration repair. Reject unrelated changes.

No model, data, loss, source policy or training exposure changes. Both students
use this shared mechanism. Qualification remains in the real production entry.
