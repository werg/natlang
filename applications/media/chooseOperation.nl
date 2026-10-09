---
description: Name the one change a media request asks for.
args:
  request: string
  source: SourceFacts
  note: Untrusted<string>
returns: "trim" | "crop" | "scale" | "transcode" | "unsupported"
---
Decide which one operation the request asks for.

1. Read request and note, and name the change the user wants in one phrase.
2. Match that change to exactly one operation:
   - "trim": shorten the clip in time (keep a span of it).
   - "crop": cut the frame down to a region of the picture.
   - "scale": change the pixel size of the picture.
   - "transcode": re-encode into another video codec or container, with the same picture and the same length.
3. Answer "unsupported" when the change is none of these, or needs two or more of them together.
