# Testing the first-run experience

Start from the repository root after `scripts/setup_dev.sh --node-only`:

```sh
natlang dashboard --root examples --open
```

Choose **hello** under Programs and click **Run**. The first run may need model
setup; its download and startup appear in Log. Use Models to choose a provider
instead. Output shows the label; Live execution shows the active lambdas.
Expand a lambda and one of its steps to see how it arrived at the result.
Try triage next to process the bundled support tickets.

For a usability session, ask a developer unfamiliar with natlang to think aloud.
Avoid giving commands unless they get stuck, and record each intervention. Use
a separate empty workspace for their own programs so examples stay unchanged.
Do not change or lower the model's reasoning settings mid-session: record the
model/profile used so failures and delays can be interpreted.

Give these tasks in order:

1. Install natlang and run a supplied example. Record installation/download time
   separately from execution and time to the first visible result.
2. Run the example on their own input. Ask where they expect input paths to resolve.
3. Create a small classifier and change its behavior. Record time to their first
   successful edit, unexpected files/concepts, and whether they discover `dev`.
4. Save an input and expected result, change the instructions, and verify that
   tests reveal the behavior change. Ask what passing tests establish.
5. Recover from a missing input file or an unavailable model endpoint. Record
   whether they locate the cause and next action without assistance.

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
