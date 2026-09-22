---
args:
  request: Request
  source: Clip
  files?: Dict<File>
returns: Plan
---
Choose exactly one of trim, crop, scale, or transcode from the user's request. If the request names a sidecar note, inspect only that args/files leaf before choosing.
Write a Plan record with exactly kind, input, output, start, end, x, y, width,
height, and keep_audio. Copy request.input and request.output exactly. For trim,
use start/end seconds in source duration. For crop, use x/y and desired width
and height within the source frame. For scale, set positive target width/height.
Use zero for unused numeric fields. Keep audio when the request does not ask to
remove it and the source has audio. Do not claim to understand an unsupported
operation by silently selecting a different one; write kind "unsupported".
