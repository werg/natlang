---
description: Decide whether two organization records refer to the same real-world vendor or customer entity.
args:
  left: EntityRecord
  right: EntityRecord
returns: boolean
readout: decision
---
Decide whether left and right represent the same real-world organization despite differences in legal suffixes (Inc., LLC, GmbH, Ltd.), abbreviations, punctuation, or subsidiary wording.
Consider name, country, taxId, and domain together: matching taxId or canonical corporate identity indicates the same entity, while conflicting tax IDs or distinct operating companies indicate different entities.
Return true when they refer to the same organization and false otherwise.
