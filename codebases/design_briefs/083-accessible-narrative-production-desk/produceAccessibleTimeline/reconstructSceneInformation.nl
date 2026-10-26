---
args: {"productions": "AccessibleProduction[]", "evidence": "EvidenceItem[]"}
returns: "SceneInformation[]"
---
For each stable scene identity recover actual dialogue, sound, visible information and source timing. Preserve the source version and evidence IDs for every fact. Distinguish essential understanding from decorative detail in the context of the narrative. Missing timeline bounds remain null; do not invent a gap or frame rate from an approximate scene description.
