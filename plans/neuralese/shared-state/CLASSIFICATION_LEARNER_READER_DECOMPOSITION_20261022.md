# Historical classification in the actual learner

The recipe verifier accepts the same historical classifier source at its original
path. The learner still required that path to equal its own runtime path, blocking
an otherwise unchanged classifier during the FFN memory repair.

1. Preserve the classifier producer file unchanged, including its strong identity.
2. Use one shared reader adapter in SFT, resident append and geometry checks.
   Authenticate historical producer SHA, byte count and exact original path.
   Require byte equality with the current classifier; validate each unique source
   once and check it again after use.
3. Persist an explicit derived reader view in the caller's output namespace.
   It changes only producer paths after authentication, records the complete
   original proof and producer bindings, and grants no training/exposure credit.
4. Run the original full migration verifier on that derived view. Retain source,
   held, parser policy, index, row-binding and topology checks without exceptions.
   The original recipe proof and earned history remain byte-for-byte unchanged.

This adapter is exact mechanism. It changes no admission policy or model-facing
input. All model lines share it; no historical producer is silently rewritten.
