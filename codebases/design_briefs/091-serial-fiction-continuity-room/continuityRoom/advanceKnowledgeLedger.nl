---
description: Reconstruct what each character knows at the proposed scene boundary.
args: { edition:SerialEdition; proposal:SerialScene }
returns: KnowledgeEvent[]
---
Order approved scenes by episode and scene order. For each secret, add a character-specific knowledge event only where the scene explicitly exposes that secret to that character. Do not infer awareness from reader knowledge, implication, or another character's private scene. Bind every event to scene and edition revision. Invalidate later knowledge states when an earlier exposure changes.
