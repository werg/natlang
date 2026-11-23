---
description: Judge whether a .nl parameter carries untrusted external input that should be typed as Untrusted<T>.
args:
  paramName: string
  paramType: string
  body: string
returns: boolean
readout: decision
---
Decide whether the parameter named paramName (currently typed as paramType) carries external, user-authored, or third-party text (such as emails, support tickets, PR diffs, web pages, uploaded documents, or chat messages) that could contain prompt injections and should be wrapped in Untrusted<...>. Return false if paramType already uses Untrusted<...>, or if the value is trusted internal configuration, schema, rubric, or code.
