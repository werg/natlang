export type MediaRequest = { text: string, input: string, output: string };
export type Clip = { id: string, status: "ok" | "failed", width: number, height: number, duration: number, has_audio: boolean,
  video_codec: string, sha256: string, detail: string };
/** The containers and video codecs the render service supports (a closed set; MEDIA_ENCODERS in index.ts maps them to FFmpeg). */
export type Container = "mp4" | "mkv" | "webm" | "mov";
export type VideoCodec = "h264" | "h265" | "vp9" | "av1";
export type Operation = "trim" | "crop" | "scale" | "transcode" | "unsupported";
export type Plan = { kind: Operation, input: string, output: string, start: number,
  end: number, x: number, y: number, width: number, height: number, keep_audio: boolean, container: Container, video_codec: VideoCodec };
export type Receipt = { status: "ok" | "failed" | "unknown" | "unsupported", output: string, exit_code: number, sha256: string, detail: string };
export type Inspection = { status: "ok" | "failed" | "unavailable", width: number, height: number, duration: number,
  has_audio: boolean, video_codec: string, sha256: string, visual_status: "supported" | "contradicted" | "uncertain" | "unavailable" | "not-needed",
  visual_detail: string, detail: string };
export type Assessment = { intent_met: boolean, explanation: string };

// What the natural-language stages see: measurements and parameters, never file names or hashes.
/** The source clip as the planning stages read it. */
export type SourceFacts = { width: number, height: number, duration: number, has_audio: boolean };
export type TrimTimes = { start: number, end: number };
export type CropRectangle = { x: number, y: number, width: number, height: number };
export type ScaleSize = { width: number, height: number };
export type TranscodeTarget = { container: Container, video_codec: VideoCodec };
/** The plan as the assessment reads it: the operation and its parameters. */
export type PlanFacts = { kind: Operation, start: number, end: number, x: number, y: number, width: number, height: number,
  keep_audio: boolean, container: Container, video_codec: VideoCodec };
/** What the inspection measured on the rendered clip. */
export type Observed = { width: number, height: number, duration: number, has_audio: boolean, video_codec: string,
  visual_status: Inspection["visual_status"], visual_detail: string };
