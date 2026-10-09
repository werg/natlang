---
args:
  task: string
  first: string
  second: string
returns: 'first' | 'second' | 'equal'
---
Two executions of the same function are described in first and second. task gives the function's instructions, its signature, the inputs of this call, and any feedback recorded about this call.

Decide which execution carries out the instructions better for these inputs. Judge by the instructions and the inputs: the right result, and the right effects (service calls, file changes, written variables). An execution that performs an effect the instructions do not ask for, or skips one they need, is worse. A service call whose arguments differ only in how a value is written (999 or "999") is the same call. Do not prefer an execution for being longer or more elaborate. Answer "equal" when both are acceptable and neither is better.
