---
description: Reconstruct joint inheritance beliefs and design a stock-feasible adaptive experiment policy.
kind: directory-reducer
args:
  request: LaboratoryRequest
returns: LaboratoryCertificate
---
1. Call readBank(folder, request). Retain the returned source inventory, interpreted pedigree,
   theories, observations, assays and material ledger as separate typed contributions.
2. Call reconstructBelief(source). Consume its actual surviving weighted worlds, rejected tickets
   and observation assessments. An unresolved likelihood remains explicit and blocks a numerical
   optimality claim until its source interpretation is resolved.
3. Call predictExperiments(source, posterior). Its actual world predictions and observable answer
   partitions are the planning input. An experiment's name does not establish its information value.
4. Call planPolicy(source, posterior, predictions, request). The returned policy carries separate
   branch-local stock and posterior states, then consumes its actual bottom-up candidate values.
5. Call auditCertificate(source, posterior, predictions, policy, request). Consume the returned
   certificate rather than reconstructing a preferred answer at the root. Preserve its unresolved items.
6. Return that certificate. Original accession, theory, experiment, material and clause IDs retain
   their source spelling. Generated world/node IDs may vary consistently. Rational values use positive
   denominators; order does not matter for worlds, witnesses or sibling answer branches.
