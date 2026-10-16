# Resident mandatory source adoption

Owner instruction: admit usable complete rows promptly and keep every admitted long
row mandatory without another model/optimizer reload. This extends the existing
resident data transaction; it does not introduce another progress or optimizer.

1. The supervisor chooses admitted sources and prepares an immutable recipe. This
   remains natural-language operational policy; no new scheduling decision lives
   in the verifier.
2. The existing crisp preparation verifier authenticates a pending declaration,
   its complete source/evidence closure and every old row binding/quota. It permits
   pending-only or pending plus exact active/held append, while retaining fixed
   foundation, runtime, numerical settings and protection checks.
3. The same update-boundary resident commit atomically publishes the complete
   recipe/plan/declaration identity. A crash before commit leaves the old generation;
   a crash after commit recovers the exact new generation before another update.
4. The trainer loads the verified declarations into its existing checkpoint options
   and merges them through native_continuation._retain_pending_sources. New declared
   rows earn zero visits; activation keeps original IDs and all existing credits.
5. Existing checkpoints and readiness consume the same required_visits,
   exposure_counts and pending_source_declarations frame. Held rows remain held.
   No extra model forward, optimizer step, RNG change or checkpoint format exists.

Qualification covers pending-only, combined active/held append, zero earned credit,
old dependency/quota preservation, mutation/reclassification rejection, and crashes
before/after the sole commit point. Independent review precedes deployment.

Qualification: 67 focused tests passed (26.40s), including actual tiny SFT updates, full parameter/optimizer/RNG equality to the all-data baseline, post-commit crash/recovery, pending-only transactions, and preserved quota activation. A quota above the current epoch count remains executable through the existing exposure scheduler, rather than becoming an unschedulable readiness deficit.

Independent root review identified a pre-commit withdrawal gap. Preparation now uses persisted actual withdrawal history and the unchanged bound proof; the live manager independently uses the restored SFT history before commit. Negative tests cover both active and pending reintroduction, including a stale CPU history that cannot bypass the live verifier. Final four-file focused run: 69 passed (24.34s); quick passed.
