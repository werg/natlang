---
args: { assets: InstalledAssetVariant[], manuals: VariantManual[] }
returns: AssetManualMatch[]
---
Match every manual's variantScope against the installed asset variant, not just its model family. Include only sections whose applicability names that installed variant. Missing or multiple equally plausible manuals remain unresolved for that asset.
