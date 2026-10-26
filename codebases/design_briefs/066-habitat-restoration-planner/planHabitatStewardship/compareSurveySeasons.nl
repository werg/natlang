---
args: { surveys: HabitatSurvey[], baseline: HabitatObservation[] }
returns: SurveyComparison
---
Pair surveys by area, indicator, method, and season. Mark different-season observations incomparable unless a supplied protocol supports adjustment. Preserve original counts and record seasonal/method confounders rather than interpreting raw differences as change.
