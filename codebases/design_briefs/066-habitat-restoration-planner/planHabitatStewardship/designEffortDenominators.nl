---
args: { objectives: RestorationObjectiveRecord[], surveys: HabitatSurvey[], samples: VolunteerSample[], comparison: SurveyComparison }
returns: MonitoringDesign
---
Map each objective to its named indicators and survey rounds. Require effort duration and observer count alongside volunteer observations. Retain only repeat methods and seasons that support comparison; show warnings where effort or season differs.
