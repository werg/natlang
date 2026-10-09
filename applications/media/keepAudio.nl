---
description: Whether the output keeps the audio track.
args:
  request: string
  source: SourceFacts
returns: boolean
---
Decide whether the output keeps the sound.

1. When source.has_audio is false, answer false.
2. When request asks to remove, mute or drop the sound, answer false.
3. Otherwise answer true.
