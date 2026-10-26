---
args: { asset: InstalledAssetVariant, manual: VariantManual }
returns: VariantManualFinding
---
First compare the exact model and installed variant with manual.model and variantScope. A model-family match alone does not establish applicability. For a matching manual, examine each section's applicableVariantIds: retain sections explicitly covering this installed variant, reject sections naming other variants, and leave empty or conflicting applicability unresolved instead of labeling it universal.

Require a named manualId and sourceVersion before choosing an edition. Return VariantManualFinding { manualId: manual.manualId, applicable: the model and variant match with an identified source edition, relevantSections: exactly the supported section IDs, blockers: missing edition, variant ambiguity or unresolved section scope }. A manual with no supported sections remains blocked for handover. This finding does not assert that facilities can open the format; that is handled by the actual export-profile check.
