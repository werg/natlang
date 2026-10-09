# Call records, compilations and specialization

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** What the runtime records for every call, the `recording` exclusions, and compilations: ordered crisp cases that may serve a call in place of the agent, and how they are promoted or demoted.

The runtime records every call: the definition and its revision, the parent call
and the eval that started it, exact inputs and captures, the result, each service
call with its arguments and result, folder changes, the eval programs run, and
cost. Values beyond a bound are recorded by hash and type; a program may exclude
definitions or arguments, which are then recorded by type only.

A call may be served by a compilation of its definition's revision: an ordered
list of crisp cases, each a guard over the call's arguments and a body with the
function's signature, run with the function's context and services. The first
active case whose guard admits the arguments serves the call. A case that fails
or returns a value of the wrong type does not fail the call: the agent runs it,
told which effects the case already performed. A compilation never changes the
program's source, applies only while the definition's context interface is the
one it was compiled against, and is promoted or demoted by measured comparison
with the agent ([TRACE_SPECIALIZATION.md](../../plans/TRACE_SPECIALIZATION.md)).
