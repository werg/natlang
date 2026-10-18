# Native evaluation external payload storage

The asynchronous evaluator keeps its request, queue, state and logical incoming
weight path on the configured evaluation filesystem. An optional payload root
stores the large verified export on another filesystem while the logical weight
path is a symlink to that content-addressed payload.

| Unit | Representation | Reason |
| --- | --- | --- |
| Which export is fetched | Existing private receipt and active transfer record | The receipt remains the authority for role, step, byte count and SHA; interrupted work still pins the same receipt. |
| Where transfer bytes live | Optional operator-configured external payload root, named by SHA | Keeping the transfer outside tmpfs avoids a second 24 GB allocation there. The root is storage configuration, not a model-specific policy. |
| Resume and verification | Existing `.pending`/aria2 control pair and final SHA verifier | The partial payload and transfer control must remain together; only a fully verified payload is exposed through the logical path. |
| Logical path publication | Temporary absolute symlink followed by same-filesystem `os.replace` and directory fsync | `native_async` and task-eval requests retain their existing path shape. The symlink itself moves into the sealed request, so no cross-device copy returns the payload to tmpfs. |
| Capacity admission | Separate logical-filesystem reserve and external free-space floor | The external filesystem must retain at least 20 GiB after the missing payload bytes are allocated. A legacy partial in the logical directory requires an explicit controlled migration, never a second implicit transfer. |
| Retention | Content-addressed payload is retained after request completion | Request cleanup unlinks its symlink only. A payload is removed only by an explicit retention action after proving no active request or other logical reference uses it. |

No model-facing instruction or policy changes. The fetcher and filesystem
operations are crisp transport and durability mechanisms.
