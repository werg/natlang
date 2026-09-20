export type PackageRequest = { name: Text, range: Text, engine: Text, purpose: Text };
export type Pin = { name: Text, version: Text, sha256: Text };
export type Lock = { root: Text, engine: Text, packages: Pin[], id: Text };
export type Resolution = { status: Text, lock: Lock, alternatives: Num, detail: Text };
export type InstallCheck = { ok: Bool, revision: Text, detail: Text };
export type InstallReport = { status: Text, target: Text, revision: Text, detail: Text };
