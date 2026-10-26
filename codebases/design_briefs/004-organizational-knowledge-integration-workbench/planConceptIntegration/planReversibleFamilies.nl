---
args: {"bridges": "MeaningBridge[]", "handoffs": "RoleHandoff[]", "scenarios": "IntegrationScenario[]", "inquiry": "IntegrationInquiry", "prior": "IntegrationAtlas | null"}
returns: "ReversibleWave[]"
---
Group record families only when their current meaning mappings, ownership and dependency scenarios fit together. Bind tests = checkRoundTripMeaning(familyScenarios, bridges), with familyScenarios the actual related IntegrationScenario values. Require source identity preservation and explicit reverse transformations. Check supplied integrity receipts against this family, exact source/mapping versions and validator, with no missing references or duplicate identities. Missing receipts or failed semantic round trips hold the family. Defer a nonrepresentable distinction until an explicit target extension is agreed. Preserve historical decisions and completed wave receipts.
