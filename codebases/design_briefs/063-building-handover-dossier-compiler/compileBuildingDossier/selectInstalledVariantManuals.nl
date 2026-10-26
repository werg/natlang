---
args: { assets: InstalledAssetVariant[], manuals: VariantManual[] }
returns: AssetManualMatch[]
---
For each installed asset, call inspectVariantManual(asset, manual) for every supplied manual in its model family. Use the actual findings to exclude wrong variants and preserve unknown section applicability. Exactly one supported manual edition with an unambiguous section set can supply manualId and relevantSections. Multiple plausible editions require source-version confirmation; do not choose the lexically newest sourceVersion.

Return one AssetManualMatch for every installed asset. Copy its assetId and variant, the uniquely verified manualId or null, only that manual's applicable section IDs, and every unresolved manual/variant/edition/section blocker. An empty catalog produces a missing-manual blocker for the asset, never an absent result row. Preserve all input manuals unchanged.
