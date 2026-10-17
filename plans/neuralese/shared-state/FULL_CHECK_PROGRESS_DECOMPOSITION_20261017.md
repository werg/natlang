# Full-check progress diagnostics

The 2026-10-17 full check stopped producing output for 25 minutes in the
CUDA attention-carrier backward test. Repeated process samples showed unchanged
CPU and I/O counters. An explicit abort with core dumps disabled revealed an
autograd worker waiting on an Inductor compilation future. The same CUDA test
and its complete file passed with fresh caches in isolation; the suite-order
or compiler-pool cause remains under investigation.

1. Existing supervision policy decides when a stalled job needs investigation.
2. The check script is a crisp process launcher. Full Python suites print each
   test name so their log identifies the current operation before it finishes.
3. Pytest's existing faulthandler prints thread stacks after five minutes in a
   test. This is diagnostic output, not a timeout, termination or skipped test.
4. Existing step accounting still records every failure and runs all steps.
   Quick checks retain their compact output. No production kernel or numerical
   tolerance changes, and a passing isolated test does not qualify a full run.

Verbose test names can themselves contain literal failure text as parameter
values. Failure routing parses pytest's anchored `FAILED` records, rather than
matching such text inside a passing test name. This is an exact log-parser
correction; test selection, verdicts and routing policy remain unchanged.
