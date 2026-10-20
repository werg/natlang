# Check-local checksum cache

Checks hash temporary fixtures through the same exact helpers as production.
Their disposable identities must not accumulate in the supervisor's shared
cache: 8,526,772 test entries enlarged that cache to 2.29 GB and caused actual
1 GiB metadata jobs to fail while loading its index.

1. Use check-main's existing temporary directory for its default checksum log.
2. Export that path to every check and its children. Preserve an explicitly
   selected caller cache, including tests that deliberately exercise a cache.
3. Remove the default log with the existing temporary-directory exit cleanup.

This is environment and disposable-file plumbing. Digest algorithms, corpus
bindings, verification and production caches do not change. The actual quick
check exercises the runner; the full-check supervisor adopts the updated runner
for its next complete suite.
