export type PackageRequest = { name: string, range: string, engine: string, purpose: string };
export type Pin = { name: string, version: string, sha256: string };
export type Lock = { root: string, engine: string, packages: Pin[], id: string };
export type Resolution = { status: string, lock: Lock, alternatives: number, detail: string };
export type InstallCheck = { ok: boolean, revision: string, detail: string };
export type InstallReport = { status: string, target: string, revision: string, detail: string };
