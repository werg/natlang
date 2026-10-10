---
kind: directory-reducer
args:
  request: ComponentRewriteRequest
returns: string
---
Improve the selected instruction components using the supplied training feedback. This is a typed edit of the component source folder.

Read components.json. Each value is either {kind:'program.guidance',text:string} or {kind:'lambda.instructions',template:{segments:string[],slotIds:string[]}}. Edit the components named in request.keys, and write each revised value in the same shape as its current one. request.feedback is a pageable read-only view: inspect request.feedback.length and request.feedback.page(start, limit) (at most 200 records per page). Select useful evidence without discarding the rest. Use it to identify a specific failure and repair its instructions concisely, and preserve the behavior the examples expect and the immutable component contracts.

Write the complete revised candidate to components.json and return one sentence naming the failure the edit addresses. When request.problem is present, your previous components.json was rejected for that problem: read the current components.json and revise it so that the problem no longer holds.
