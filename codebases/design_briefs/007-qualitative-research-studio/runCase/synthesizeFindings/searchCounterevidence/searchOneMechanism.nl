---
args: { code: CodeBoundary, mechanism: string, evidence: EvidencePage[] }
returns: CodeMechanismSearch
---
For every supplied passage that could support or narrow this exact code mechanism, bind judgment = classifyCounterexample(code, mechanism, passage). Keep source text direction and participant context distinct. Return supporting and contradictory references separately; when a contradiction changes the scope, return the narrower interpretation. Preserve missing partitions and the specific search question.
