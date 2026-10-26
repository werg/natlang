---
description: Route one serial-identified returned device through evidence, safe repair, custody, and resale controls.
args:{request:DeviceHandlingRequest;lifecycle:DeviceLifecycleFile}
returns:DeviceDispositionProposal
---
Bind scope = `frameDeviceReturn(request, lifecycle.returns)` and policy = `selectDeviceHandlingPolicy(lifecycle.purchaseTerms, lifecycle.approvedHandlingRules, lifecycle.marketplaceRules)`. Preserve customer, item, purchase, serial and case version; an ambiguous identity stops transfer. Bind condition = `assessDeviceCondition(scope, policy, lifecycle.conditionReports, lifecycle.imageAnnotations, lifecycle.serialLedger)`. Separate cosmetic observations, functional symptoms and unverified allegations; ask only for evidence that changes routing.

Bind disposition = `selectSafeReturnPath(scope, policy, condition, lifecycle.conditionReports, request.preferredOutcome)`. Apply original purchase terms and current approved handling rules. If diagnosis or entitlement is incomplete, keep specialist inspection pending and do not promise remedy. If repair is selected, bind repair = `planAuthorizedRepair(scope, policy, disposition, lifecycle.partCompatibilityReceipts, lifecycle.workshopCapacityReceipts)`; require compatible exact part/model/revision, approved procedure and a current reserved slot. A substitution without compatibility proof stays unresolved.

Bind custody = `reconcileSerialCustody(scope, policy, disposition, repair, lifecycle.serialLedger, lifecycle.custodyReceipts)`. Treat collection, intake, repair transfer and return as separate observed handoffs. A serial mismatch blocks movement; preserve new transit damage as its own dispute and update inventory only from a confirmed receipt.

Only after completed required functional tests, current title checks, and passed device-specific erasure, bind resale = `prepareVerifiedResale(scope, policy, custody, lifecycle.erasureVerification, lifecycle.functionalTestReceipts, lifecycle.marketplaceRules)`. Quarantine residual data or unknown erasure. Disclose every known defect in marketplace-approved grade terms. If a return label expires, issue a new logistics artifact under the existing entitlement and operation identity. Return only verified state and blockers; physical movement, erasure and listing remain separate authorized effects.
