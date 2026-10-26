---
description: Assess distinctive reporting value and feasibility before selecting an issue portfolio.
args:{brief:IssueBrief;pitches:StoryPitch[];existing:Commission[];sectionIds:string[];publicationDate:string}
returns:PitchAssessment[]
---
Bind placements = `analyzePitchPlacements(brief, pitches, existing, sectionIds, publicationDate)`. For each pitch use its returned eligible sections, deadline conflict and dependencies to derive feasibility and riskDependencies. Compare every pitch's angle against issue goals, audience, known contributor work and reporting plan. Mark access viable, conditional or unproven from supplied evidence. Link pitches that substantially duplicate angle or reporting sources. Do not treat a pitch as an issue placement unless its actual format, word count, section fit, contributor workload and reporting deadline can coexist with already accepted commissions.
