---
description: Isolate proposed substitutions that affect an approved clinical requirement for clinician review.
args: { dossier: TransitionDossier; change: TransitionChange; taskGraph: PrerequisiteState[] }
returns: string[]
---
Compare proposed logistical substitutions with the approved plan's clinical requirement references. Identify only the affected step and exact change. Require a current clinician review tied to that plan version. Do not decide whether the substitution is clinically equivalent.
