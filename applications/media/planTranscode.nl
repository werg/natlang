---
description: The container and video codec of a transcode.
args:
  request: string
  extension: string
  source: SourceFacts
  note: Untrusted<string>
  problem?: string
returns: TranscodeTarget
---
Work out the container and video codec the clip is re-encoded into. extension is the file extension of the output
name (for example "webm").

1. Read request (and note when the request points to it) for a format or codec: "WebM", "MP4", "MKV", "MOV",
   "H.264", "H.265" or "HEVC", "VP9", "AV1".
2. Name the container: "mp4", "mkv", "webm" or "mov". When the request names a container, use it; otherwise use the
   container of extension.
3. Name the video codec: "h264", "h265", "vp9" or "av1". When the request names a codec, use it; otherwise use the
   usual codec of the container: h264 for mp4, mov and mkv, vp9 for webm.
4. mp4 holds h264, h265, vp9 and av1; mov holds h264 and h265; mkv holds all four; webm holds vp9 and av1.
5. When problem is given, it states what was wrong with the previous answer; give a target that fixes it.
