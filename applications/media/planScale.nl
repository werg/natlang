---
description: The target size of a scale.
args:
  request: string
  source: SourceFacts
  note: Untrusted<string>
  problem?: string
returns: ScaleSize
---
Work out the target size of the picture in pixels.

1. Read the target in request (and note when the request points to it): an absolute size ("1280x720"), one side
   ("width 640"), a factor ("half size", "twice as large") or a named height ("720p").
2. With one side or a named height, compute the other side from the source proportions: height = width * source.height /
   source.width, or width = height * source.width / source.height. With a factor, multiply both source.width and
   source.height by it.
3. Round width and height to whole numbers of at least 1. Do the arithmetic in eval.
4. When problem is given, it states what was wrong with the previous answer; give a size that fixes it.
