# V20 checkpoint eviction incident

This record documents an incomplete recovery snapshot. Two V20 full-state checkpoint files were removed from Pop before a remote copy was confirmed. An explicit DGX path check found neither file; exact-size/hash searches found no alternative copy. The V20 full-state optimizer state is therefore unresolved and must not be described as mirrored or recoverable.

The original expected-byte manifest is unchanged. Its two checkpoint entries remain present and missing. The nine remaining files are registered as a separate surviving-artifacts snapshot. No training admission or model qualification follows from either snapshot.

See `incident.json` for the exact expected hashes, byte counts, verification result, and impact.
