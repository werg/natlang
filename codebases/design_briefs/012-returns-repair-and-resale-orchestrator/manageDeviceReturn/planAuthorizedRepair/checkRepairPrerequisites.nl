---
description: Verify exact part compatibility and workshop task capacity before authorizing a plan.
args:{disposition:StageResultReturnDisposition;compatibility:PartCompatibilityReceipt[];capacity:WorkshopCapacityReceipt[]}
returns:{compatiblePartIds:string[];blockedPartIds:string[];capacityTaskIds:string[];unavailableTasks:string[];evidenceIds:string[];}
---
Accept a part only when receipt item, model, revision and approved procedure match the returned device and chosen path. Accept capacity only for named tasks, accepted slot and current window. Missing receipts block the task; never infer compatibility from appearance or model family.
