---
description: pi's bash tool. Run one command through the environment's shell with bounded tail output (pi-durable tools/bash.ts).
args:
  args: "{ command: string, timeout?: number }"
returns: ToolExecutionResult
---
Run args.command. Its output streams into this call's output, and the host keeps the tail that the model will see, so
a successful run returns no content of its own. When a step says "fail with", finish with status failed and that text
exactly as the reason; a failed run still shows the output and diagnostics it produced.

1. Check the timeout (seconds) when args.timeout is given:
   - not a finite number, or not greater than 0: fail with "Invalid timeout: must be a finite number of seconds";
   - greater than 2147483.647: fail with "Invalid timeout: maximum is 2147483.647 seconds".
2. run = env.runShell(args.command, args.timeout).
3. The spill file: when run.value.spillPath or run.error.spillPath is set, record
   env.diagnostic({ severity: "info", code: "full_output", message: "Full output: <spillPath>" }) before anything else.
4. The outcome, in this order:
   - run.error.code is "timeout": fail with "Command timed out after <args.timeout> seconds";
   - run.error.code is "aborted": fail with "Command aborted";
   - any other run.error: fail with run.error.message;
   - run.value.exitCode is not 0: fail with "Command exited with code <exitCode>";
   - otherwise return {} (the retained output becomes the content).
