export type MediaRequest = { text: string, input: string, output: string };
export type Clip = { id: string, status: "ok" | "failed", width: number, height: number, duration: number, has_audio: boolean,
  sha256: string, detail: string };
export type Plan = { kind: "trim" | "crop" | "scale" | "transcode" | "unsupported", input: string, output: string, start: number,
  end: number, x: number, y: number, width: number, height: number, keep_audio: boolean };
export type Receipt = { status: "ok" | "failed" | "unknown" | "unsupported", output: string, exit_code: number, sha256: string, detail: string };
export type Inspection = { status: "ok" | "failed" | "unavailable", width: number, height: number, duration: number,
  has_audio: boolean, sha256: string, visual_status: "supported" | "contradicted" | "uncertain" | "unavailable" | "not-needed",
  visual_detail: string, detail: string };
export type Assessment = { intent_met: boolean, needs_visual_review: boolean, explanation: string };
