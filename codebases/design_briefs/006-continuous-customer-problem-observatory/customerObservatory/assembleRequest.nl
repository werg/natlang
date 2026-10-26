---
args: { question: ObservatoryQuestion }
returns: CustomerObservatoryRequest
---
Map only the question's stated case, target population, interval, requester, and snapshot version into the observatory request. Keep the studyQuestion text intact; never derive a rate denominator from the wording.
