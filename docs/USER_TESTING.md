# Testing the first-run experience

Rehearse and run each session on a clean machine, container or VM with only Node.js 22.13 or newer installed: a
developer machine hides installation problems. Install the way a new user would:

```sh
curl -fsSL https://raw.githubusercontent.com/werg/natlang/main/scripts/install.sh | sh
```

(To test unreleased changes, install from a clone instead: `git clone https://github.com/werg/natlang.git ~/natlang &&
~/natlang/scripts/setup_dev.sh --node-only`.) Participants then work in an empty directory, for example
`mkdir ~/natlang-session && cd ~/natlang-session`, outside any clone, so nothing in a checkout guides them.

If the clean install itself fails and the session must go on, fall back to the development path: from the clone's
root, `natlang dashboard --root examples --open`, choose **hello** under Programs and click **Run**. Output shows the
label; **Execution** shows the active lambdas, and expanding a lambda and one of its steps shows how it arrived at the
result. Record the fallback and the install failure as a finding.

Choose the model before the session and keep it for every participant in a round. A hosted model (a Pi provider
or an OpenAI-compatible endpoint, configured on the Models page) tests the language, CLI and dashboard without
model latency dominating the session. The managed local model is a separate track: on a CPU-only machine its
first call can take minutes (a 1.8 GB download, then a prompt of several thousand tokens to prefill), so time
it on the participant's hardware and report it apart from the authoring tasks. A new program's `answer.nl`
uses `readout: decision`, so the model scores its labels instead of writing an answer: keep it for the timed first
run (it took about 19 s instead of 136 s on the local LFM3 1.7B). A participant's own function that writes free text
or code takes as long as the model needs on that hardware; time it with a hosted model unless the local track is the
subject.

For a usability session, ask a developer unfamiliar with natlang to think aloud.
Avoid giving commands unless they get stuck, and record each intervention.
Do not change or lower the model's reasoning settings mid-session: record the
model/profile used so failures and delays can be interpreted.

Give these tasks in order:

1. Install natlang (the clean install above), then `natlang init` and `natlang run` in the empty directory. Record installation/download time
   separately from execution and time to the first visible result.
2. Run the program on their own input (`natlang call answer.nl "..."`). Ask where they expect input paths to resolve.
3. Create a small classifier and change its behavior. Record time to their first
   successful edit, unexpected files/concepts, and whether they discover `dev`.
4. Save an input and expected result, change the instructions, and verify that
   tests reveal the behavior change. Ask what passing tests establish.
5. Recover from a missing input file or an unavailable model endpoint. Record
   whether they locate the cause and next action without assistance.

## Coding-agent sessions

Many participants will build with a coding agent rather than by hand. Run the same five tasks with the
participant directing Claude Code or Codex in an empty workspace, after `natlang skills install`. The agent
works without a terminal of its own, so every step must succeed non-interactively: `natlang init`, the
`AGENTS.md` it writes (`CLAUDE.md` imports it), `natlang check --json`, `natlang call answer.nl "..." --json`,
`natlang test --json`, `natlang traces show CALL_ID` (the ID `natlang call` prints on stderr), `natlang traces fix
CALL_ID --expected JSON`, and the missing-model error (exit code 3, naming `natlang setup --yes` or a hosted model).
Exit codes: 0 success, 1 the program is wrong, 2 usage error, 3 model or runtime unavailable. Record, in addition to the table below: the agent's first command, every command it got wrong
and the error text it saw, whether it read AGENTS.md or the skill, whether it asked before the model download,
model calls it made before `natlang check` passed, and whether it used traces or guessed when a test failed.
An agent that retries a failing command unchanged or edits generated `*.d.nl.ts` files marks a gap in the
errors or the guide; fix the message, not the agent's prompt.

At the end, ask which model ran, what source checking guarantees, what the model
can access, and how they would inspect an incorrect result. These questions
test whether the experience communicates the programming model.

Use this record for each participant:

| Measure | Result |
|---|---|
| OS, Node version, CLI revision | |
| Model/profile and machine | |
| Install and download time | |
| First result after setup | |
| First successful behavioral edit | |
| Tasks completed without help (out of five) | |
| Recovery time and assistance | |
| Incorrect assumptions and quotes | |

Run five initial sessions, group recurring problems by the step they prevented,
fix those steps, and repeat with new participants. A session is evidence of that
person's experience, not a model-quality benchmark. The automated CLI tests and
`npm --prefix ts-host run test:dashboard` check the mechanics with deterministic
model fixtures; they do not replace participant testing or live model evaluation.
