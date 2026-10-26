---
args: { request: HarmonizationRequest, harmonizations: Harmonization[], policy: PolicySelection }
returns: string[]
---
Bind target construct and intended analysis to exact dataset IDs, data owner, protocol and expected version. Check that the applicable policy identifies the same owner and scope and does not authorize dataset expansion beyond the approved protocol; preserve any unresolved authority or purpose limitation.
