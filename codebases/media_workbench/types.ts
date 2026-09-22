export type Request = { text: Text, input: Text, output: Text };
export type File = { kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num };
export type Clip = { id: Text, status: Text, width: Num, height: Num, duration: Num, has_audio: Bool, sha256: Text, detail: Text };
export type Plan = { kind: Text, input: Text, output: Text, start: Num, end: Num, x: Num, y: Num, width: Num, height: Num, keep_audio: Bool };
export type Receipt = { status: Text, output: Text, exit_code: Num, sha256: Text, detail: Text };
export type Inspection = { status: Text, width: Num, height: Num, duration: Num, has_audio: Bool, sha256: Text, visual_status: Text, visual_detail: Text, detail: Text };
export type Assessment = { intent_met: Bool, needs_visual_review: Bool, explanation: Text };
export type MediaResult = { status: Text, output: Text, technical_ok: Bool, semantic_ok: Bool, needs_visual_review: Bool, explanation: Text, receipt: Receipt, inspection: Inspection };
