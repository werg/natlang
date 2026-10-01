---
kind: directory-reducer
args:
  request: ComponentRewriteRequest
returns: Candidate
---
Improve the selected instruction components using the supplied training feedback. This is a typed edit of the component source folder.

Read components.json. Each value is either {kind:'program.guidance',text:string} or {kind:'lambda.instructions',template:{segments:string[],slotIds:string[]}}. Edit only request.keys. Keep every kind, slotIds, unselected component, and the number of segments unchanged. Static segments are ordinary string arrays: do not encode an array as a string. Preserve the behavior expected by the examples and immutable component contracts. request.feedback is a pageable read-only view: inspect request.feedback.length and request.feedback.page(start, limit) (at most 200 records per page). Select useful evidence without discarding the rest. Use it to identify a specific failure and repair its instructions concisely.

Write the complete revised candidate to components.json, then return that exact object with `return candidate;` in an eval. Do not merely inspect it. The returned value and edited file must agree.
