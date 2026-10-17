# Continue teacher generation after an engine process exits

The teacher is an outside-world service. The generation recipe keeps the same
qualified model, image, arithmetic, template, task runtime and protected TRAIN
source. A new serving process needs a new actual execution identity. No failed
or unstarted sample becomes successful because the service was restarted.

1. Authenticate the configured producer bytes and actual process identities.
   Changed immutable producer bytes are a repair error. A gone/zombie/recycled
   serving owner is an observed unavailable service, never a successful actor.
2. Let the actor-pinned feeder and collectors finish accounting before starting
   a replacement at the same endpoint. Preserve the exact original batch and
   canonical boundary; queue every unresolved or unstarted key for its existing
   independent repair. Breadth can advance with that explicit durable debt.
3. Invoke one explicitly bound engine-service command with a durable request
   before launch. Its immutable specification binds the existing engine argv,
   model/image/settings/template assets and API contract. Reconcile the same
   attempt after interruption; unknown launch state never authorizes a second
   engine. Record exit/error and log identities even when recovery fails.
4. When the replacement is live and its actual API/argv/model contract matches,
   publish a fresh serving actor/process identity. Preserve old actor evidence.
   The only generation configuration change is teacher_identity, through the
   existing canonical config revision and durable compare-and-swap. Preserve
   feeder/runtime/source/queue/cursor and ordered ready-packet authority.
5. Resume the canonical guardian, existing repair queue and next declared task
   packet. Observe actual starts/results rather than treating endpoint readiness
   as data generation or admission. Continue supervision so another observed
   engine exit does not permanently terminate the generation guardian.

The numbered policy is natural language. Crisp code is limited to the outside
service invocation, process/byte/API contract verifiers, exact launch ownership,
atomic configuration publication and existing guardian plumbing. No model-facing
prompt or task content changes. Recovery is opt-in under a bound declaration;
ordinary configurations retain their existing strict identity behavior.

The engine guard runs as a dedicated child subreaper with no unrelated existing
children. It observes the entire owned session and actual PPID descendants,
including adopted orphans that start a separate session. Signal handling keeps
the guard alive until cleanup finishes. Natural leader exit, a guard signal,
RAM danger and observer failure all drain the owned descendants, escalate after
the existing declared shutdown grace, reap orphan children and record the real
leader return code. Every signal rechecks PID/boot/start identity; a recycled
leader never authorizes signalling a replacement process. Clean mapped source
pages are distinguished from the conservative nonreclaimable RAM bound.
