import type { PackageRequest, Pin, Lock, Resolution, InstallCheck, InstallReport } from "../types.js";
import { host } from "natlang:runtime";

export default async function publish(lock: Lock, target: string): Promise<InstallReport> {
try { return await host.packages.install(lock, target); }
catch (error) { return { status: 'failed', target: target,
  revision: '', detail: String(error) }; }
}
