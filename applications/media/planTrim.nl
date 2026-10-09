---
description: The start and end seconds of a trim.
args:
  request: string
  source: SourceFacts
  note: Untrusted<string>
  problem?: string
returns: TrimTimes
---
Work out the span of the clip to keep, in seconds from the start of the source.

1. Read the times in request (and note when the request points to it). Convert them to seconds; one minute is 60 seconds.
2. For "the first N seconds": start is 0 and end is N.
   For "the last N seconds": start is source.duration - N and end is source.duration.
   For "from A to B": start is A and end is B.
   For "cut off the first N seconds": start is N and end is source.duration.
3. End is at most source.duration, start is at least 0, and start is below end. Do the arithmetic in eval.
4. When problem is given, it states what was wrong with the previous answer; give times that fix it.
