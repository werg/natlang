---
args: { request: ResearchWindowRequest, snapshot: ResearchOperationsSnapshot }
returns: ProtocolMandate[]
---
1. Bind located = locateApprovedProtocols(request, snapshot). Preserve every candidate protocol version and approval state.
2. Bind conditions = extractProtocolPredicates(located.mandates, request). Keep sample windows, duration, measurement capability, named operator, and hard scientific conditions separate from preferences.
3. Bind authority = resolveProtocolPrecedence(located, conditions, snapshot). For competing authoritative versions retain both, mark only affected experiments blocked, and emit a precedence question rather than selecting by recency or upload order.
4. Return authority as the complete protocol review, including missing and conflicting protocol IDs.
