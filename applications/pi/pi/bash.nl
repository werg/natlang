---
description: Execute a bash command in the project directory. Returns stdout and stderr. Output is truncated to the last 2000 lines or 50KB; the full output is saved to a file. purpose says what the command is for; timeout is in seconds (no limit when not given).
args:
  command: string
  purpose: string
  timeout?: number
returns: string
model: small
---
Run command for a coding agent, as pi's bash tool does, after a safety check.

First judge it: decide(risk, command, purpose). Let D be the probability of destructive and R that of review plus
D, both with two decimals.
- When D is at least 0.7, do not run it. Answer `[Refused by the safety check: this command looks destructive
  (p(destructive)=D, p(review or worse)=R). Find a safer way to do it, or ask the user to run it.]`
- Otherwise, when R is at least 0.5, ask user.confirm whether to run it, quoting command and its purpose. If the
  user does not approve, answer `[Not run: this command needs the user's approval (p(destructive)=D, p(review or
  worse)=R), and none was given. Do the task without it if you can; otherwise say in your answer which command the
  user should run and why.]`

Run it: shell.run(command, timeout). Then shape what it printed:
- More than 80 lines: save all of it to a new file under /tmp (shell mktemp), and answer what digest(its last 2000
  lines, at most 51200 bytes, purpose) writes, then `[A small model condensed N lines of output; the full output is
  in <file>]`.
- Otherwise keep the last 2000 lines and at most the last 51200 bytes. When that leaves some out, save all of it in
  the same way and add `[Output truncated; full output: <file>]`.
- Nothing printed: `(no output)`.
End with `[Timed out after T s]` when it timed out, or `[Exit code N]` when it exited with a code other than 0.
Never change the kept lines.
