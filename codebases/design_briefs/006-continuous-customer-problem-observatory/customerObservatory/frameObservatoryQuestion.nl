---
args: { request: CustomerObservatoryRequest, studies: ProblemStudy[] }
returns: ObservatoryFrame
---
Bind the observatory question to a defined population, time window, owner and study version. Keep the problem-rate denominator and data collection purpose explicit; a dashboard's case volume is not a population rate.
