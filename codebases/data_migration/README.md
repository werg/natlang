# Data migration studio

Natlang maps source columns and judges customer identity. The host checks every
mapping, uses decimal arithmetic for money, accounts for every source row, and
builds an immutable preview. Applying a preview checks source and target revisions
then uses one SQLite transaction for customers, orders, source identity mappings,
lineage and a revision increment. Uncertain identity remains a review item.
Display-name similarity never authorizes a merge. The source data stays outside
natlang state except for bounded projected rows.
