---
description: The content shown after a form was submitted - thanks, or what to fix.
args:
  purpose: Text
  review: Review
returns: Text
---
Write a short HTML fragment telling the visitor what happened to their submission (`args/review`): thank them if it
was accepted, otherwise say kindly why not (`reason`) and invite them to try again. Link back to "/".
