---
args: { artifacts: ReleaseArtifact[], challenges: EvidencePage[] }
returns: string[]
---
For each disclosure challenge, map released artifact and redaction locators back to exact reviewed source spans, policy basis and approval. Preserve immutable mapping and challenge state; never overwrite the released artifact's audit trail.
