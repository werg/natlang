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
