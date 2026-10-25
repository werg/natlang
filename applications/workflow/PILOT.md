# Selected-execution pilot

`ts-host/scripts/workflow-selected-pilot.mjs` is the engineering entry point for
the order application. It calls the compiled workflow's `WorkflowService` and
`stepFull` against a private temporary directory. The service's local receipt
file is the simulated inventory/payment/shipping boundary; no payment, delivery,
or customer-message system is contacted.

Run the selected compiled-policy journey against controlled reference
projections after building applications:

```sh
node ts-host/scripts/workflow-selected-pilot.mjs --mode controlled --seed 17
```

Run the ordinary `stepFull` reference journey separately with
`--mode reference --seed 17`.

The script compiles only `applications/workflow` through `buildProject` on first
use, so the pilot does not depend on optional application-local packages. Build
the host first (`npm --workspace @natlang/typescript-host run build:node`).
Run the focused production-interface journeys with
`node --test ts-host/test/workflow-selected-pilot.test.mjs`.

For selected checkpoint execution (point `--model-command` and arguments at the
Megatoken served-checkpoint JSONL process):

```sh
node ts-host/scripts/workflow-selected-pilot.mjs --mode model \
  --model-command /path/to/python --model-args '["/path/to/serve_selected_semantic.py","--served","/path/to/served"]' \
  --checkpoint-identity <64-hex-checkpoint-identity> --seed 17 --rpc-timeout-ms 300000
```

The report records the explicit origin, seed, checkpoint identity, every ordered
event/decision/revision/phase, local receipts, and application service events.
Controlled mode uses the production crisp implementation and production
transition/receipt mechanism. It is an engineering reference run, not evidence
of model capability or training quality.

The selected compiled-policy route is also available as `runSelectedPolicyOrder`
from the pilot module. It compiles the application's existing
`handle/crisp.ts` policy source as a callable module, initializes it through
`moduleInstanceAsync`, and invokes it under
`installSelectedSemanticEnvironment`. The selected guest returns the decision;
the host then passes that value to the production `WorkflowService.apply`. The
inventory/payment/shipping receipt is simulated locally by WorkflowService and
is outside the selected interpreter's broker. This route does not ask an NL
model to author policy code.

CLI model mode requires `--model-command`, optional JSON `--model-args`, and an
immutable 64-hex `--checkpoint-identity`. The Megatoken process is initialized
with the declared seed and checkpoint, then serves the compiled selected policy
module. A missing, invalid, or refused selection is retained in the report and
does not call `WorkflowService.apply`; there is no V8 retry. Use only local
simulated services. Model outcome quality is not measured by this pilot.

The test-only controlled selector runs the compiled selected policy across
ordinary success, a definite operation rejection,
lost-acknowledgement reconciliation after constructing a fresh service instance,
a lost request still pending after restart, cancellation compensation including
refund-before-release, and repeated events after settlement. The JSON report
includes the ordered application decisions and native service-event trace, plus
whether the temporary state directory was removed. These scenarios are
not optimizer, training, or capability evaluations.

Controlled selected-mode records are explicitly return-value projections, not
full source traces or admitted teacher labels. Each report records the original
crisp source SHA-256, a fixed projection-definition SHA-256, the exact selected
entry-source hash check, and a digest of the snapshot/limits supplied to the
owned tracer. When the owned tracer supports a call, its expected decision is
compared with an independent execution of the same compiled crisp source; its
unsupported-operation cases use that compiled-source execution directly. The
selected interpreter receives only the fixed projection records. These
engineering fixtures do not establish learned capability (`learnedSuccess`
remains unmeasured).

The fixed served bundle `c7d9040e…a0943` was exercised on CPU with seed 17,
2 OpenMP/MKL threads, and a 16-token output bound. It refused the first selected
module-initialization request because the published source exceeded the
checkpoint's declared history. The application therefore recorded a rejected
journey with zero simulated effects; the model process closed gracefully and
flushed its serving profile. This records the real current result and does not
claim successful policy execution. The run report and serving profile are
preserved as separate validation artifacts alongside the repository checkout.
