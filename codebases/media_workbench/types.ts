export type Request = { text: string, input: string, output: string };
export type File = { kind: "text", text: string, bytes: number } | { kind: "binary", bytes: number };
export type Clip = { id: string, status: string, width: number, height: number, duration: number, has_audio: boolean, sha256: string, detail: string };
export type Plan = { kind: string, input: string, output: string, start: number, end: number, x: number, y: number, width: number, height: number, keep_audio: boolean };
export type Receipt = { status: string, output: string, exit_code: number, sha256: string, detail: string };
export type Inspection = { status: string, width: number, height: number, duration: number, has_audio: boolean, sha256: string, visual_status: string, visual_detail: string, detail: string };
export type Assessment = { intent_met: boolean, needs_visual_review: boolean, explanation: string };
export type MediaResult = { status: string, output: string, technical_ok: boolean, semantic_ok: boolean, needs_visual_review: boolean, explanation: string, receipt: Receipt, inspection: Inspection };
