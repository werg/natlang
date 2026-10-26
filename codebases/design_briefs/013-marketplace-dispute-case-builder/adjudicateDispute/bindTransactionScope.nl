---
args: { request: DisputeBuildRequest, cases: DisputeCase[] }
returns: DisputeScope
---
Match exactly one dispute case by case ID, transaction ID, and both named parties. Preserve expected version, disputed outcome and disclosure purpose. Do not merge records on names or listing similarity; mismatched party or transaction identity stays blocked.
