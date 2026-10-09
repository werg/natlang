---
description: Whether a request depends on what the picture shows.
args:
  request: string
returns: boolean
---
Decide whether the request depends on what the picture shows.

1. Read request and look for a subject, face, object or composition that must be in a place in the frame.
2. Answer true for such a request, and false for a request about time, size, sound or format only.
