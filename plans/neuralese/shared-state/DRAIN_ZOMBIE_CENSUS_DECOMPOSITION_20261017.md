# Distinguish finished children from a growing optimizer process tree

The supervisor chooses the safe stop boundary. The shared census verifies which
live process identities belong to the actual optimizer before that signal; a
finished, unreaped child cannot own continuing optimizer work.

1. Discover descendants through both all-thread child lists and actual PPIDs.
2. Retain the initial traversal's existing exclusion of finished Z/X processes.
3. Apply the same exclusion to final consistency checks. For an untracked Z/X
   entry in the final PPID snapshot, observe it again before excluding it: a PID
   recycled into a live process must still reject the census.
4. Reject every new live descendant and all existing parent/identity changes as
   before. Keep the actual owner required live. Do not publish an intent or send
   a signal when the census is inconsistent.

This is a narrow exact Linux observation fix. No change to ownership policy,
model math, training state, source admission or the recorded handoff schemas.
Qualify an actual unreaped zombie beside a live descendant, with and without
thread child-list support, and simulated live growth/PID reuse in both final
discovery paths.
