---
description: Index selected media time ranges with context and rights provenance.
args:{assets:MediaAsset[];cuts:Cut[]}
returns:{cut:Cut;asset:MediaAsset;durationFrames:number;context:string;permissionIds:string[];}[]
---
For every cut resolve exactly one asset at the recorded revision. Reject out-of-range or reversed frame values. Preserve the recorded claim, qualifier and neighboring context without turning testimony into independently established fact. Carry permissions and source date on every selected range.
