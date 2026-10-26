---
args: { dossier: EngineeringChangeDossier }
returns: EngineeringChangePlan
---
1. Bind applicability = partitionByApprovedConfiguration(dossier.change, dossier.baselineConfiguration, dossier.dependencyEdges). Evaluate each unit and serial range against its own approved revision; do not propagate a blanket replacement instruction.
2. If applicability.documentationOnly is true, prepare controlled reference-number updates and preserve the existing operational configuration without initiating physical retrofit.
3. Bind stock = dispositionRevisionStock(dossier.inventory, dossier.supplierShipments, applicability). Old and new revisions may coexist, but each allocation must name the exact compatible configuration; shipped old revision gets an explicit disposition decision.
4. Bind retrofit = reconcileUnitRetrofitStates(dossier.retrofitProgress, dossier.fieldedUnits, applicability). Keep partially changed units fenced in their mixed configuration until every required verification passes.
5. Bind downstream = propagateAffectedInterfaces(applicability.affectedDependencies, dossier.evidence). Preserve unaffected documents, interfaces, and grandfathered maintenance guidance.
6. Return { changeId: dossier.change.id, applicability, inventory: stock, retrofit, affectedDownstream: downstream, verificationRequirements: retrofit.nextStepsByUnit, planDisposition: applicability.unresolvedConfigurations.length > 0 || stock.unresolvedItems.length > 0 ? "needs-review" : "proposed", commitAllowed: false }.
