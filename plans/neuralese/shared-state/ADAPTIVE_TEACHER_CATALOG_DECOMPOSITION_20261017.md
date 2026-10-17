# Continue the useful teacher task catalog

The catalog discovers already admitted trajectories that contain executable
task programs. It is candidate discovery, not data admission or generation.
Its current registry-wide freeze prevents useful continuation whenever another
source is admitted. Keep completed source work and make the authority change
explicit instead.

1. Reuse the existing catalog's exact bound extraction policy, compressed
   storage, source identities and committed row cursors. Bind the current
   registry and protection separately from the historical scan.
2. Discover the current admitted source population. Registry additions do not
   invalidate unchanged files. A withdrawn source is excluded from the current
   population; it does not erase its historical source work. Changed source
   bytes require investigation rather than silently reusing their old cursor.
3. Compare protection group sets and decision/source-review authority. If they
   change, audit the affected original source prefixes under the current
   policy. Audit cursors are separate from completed extraction cursors, so an
   interrupted policy audit never loses source progress. A general decision or
   family-policy change audits every current prefix. Never reinterpret old
   reasons as new approval.
4. Continue incomplete source files and newly admitted files. Save source and
   policy progress together in the SQLite transaction. Checkpoint publication
   records the exact authority and scanner. Authority changes during a scan
   schedule another continuation; they do not discard committed work.
5. Candidate publication requires a completed current audit and current source
   population. It still requires reference execution and fresh protection
   checks before generation or training admission. Report historical versus
   current evidence explicitly. Only current sources grant eligibility or
   provenance; historical held/quality/nontrain occurrences and identity
   conflicts remain barriers until explicitly resolved. Registry withdrawal
   never makes protected content usable. Publish compressed candidates and
   use the existing packet/reference mechanism, with this same eligibility
   rule and a bound current source population.

These numbered instructions are the policy. Crisp code implements SQLite
transactions, source hashes, immutable authority records, exact prefix
validation, discovery plumbing and the existing extraction predicate. There
is one scanner and one extraction policy, shared by both students. Neither
student training nor the running teacher is restarted for catalog maintenance.

## Committed audit reuse and exact hold lookup (2026-10-18)

1. The historical extraction policy remains the policy. An independent reviewer may
   authorize reuse of a specific interrupted audit through a SHA-bound witness
   binding its exact authority, intent, audit sidecar, current scanner and static
   dependencies. This is a verifier of an explicit review, not a policy exception.
2. Reuse requires exact held/quality/pending sets, decision holds, source-review
   identity and historical extraction mechanism. Registry/population changes do
   not grant source membership; current discovery still controls positive source
   provenance. Withdrawn-path audit debt remains durable. Real decision changes
   invalidate reuse. Subsequent registry/population-only changes carry committed
   audits automatically only when scanner, every static dependency, hold lookup
   and all decision authority remain byte-identical. The catalog lease authenticates the old namespace before
   copying committed cursors; extraction cursors and SQLite rows are never reset.
3. A completed audit prefix needs only a fresh complete source SHA/size check.
   It does not need to decode the same JSON rows again. Cursor bounds remain strict.
4. The repeated exact held-program set comprehension may be hoisted only when the
   authenticated historical reason_for AST has that precise assignment. All other
   function AST remains unchanged, and the optimized function is bound to one
   immutable holds object for the invocation. This is exact execution plumbing;
   it cannot alter eligibility policy.
5. Candidate extraction, reference qualification, teacher generation, admission and
   training exposure remain separate. No witness grants admission or readiness.
