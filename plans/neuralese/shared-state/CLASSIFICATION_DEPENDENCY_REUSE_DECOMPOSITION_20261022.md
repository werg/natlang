# Reuse authenticated dependencies within one enumeration

The supervisor observed an idle training GPU while the controller enumerated
historical classification dependencies during a stage handoff. Policy remains
the existing natural-language requirement: retain every actual source, parser,
held-data and publication dependency, and reject changed authority.

1. The existing recipe and publication stages choose which proof to enumerate.
2. The crisp verifier authenticates each declared file against its exact SHA and
   optional byte count, recording its resolved path and filesystem signature.
3. Repeated occurrences reuse that authentication only within this call, after
   checking the declaration, path and signature again. Completed proof subtrees
   are traversed once; an ancestor occurrence still reports a cycle.
4. Before returning, check every file's path and signature again. A later call
   starts with no certificates and authenticates its dependencies anew.

This is exact verification plumbing, not a new admission, scheduling or training
policy. Returned dependency bindings and ordering remain unchanged. Test shared
dependencies, contradictory declarations, mutations during enumeration,
replacement and symlink retargeting, plus independent subsequent calls.
