import type { PackageRequest, Pin, Lock, Resolution, InstallCheck, InstallReport } from "../types.js";
import { host } from "natlang:runtime";

export default function solutions(request: PackageRequest): Lock[] {
return host.packages.solutions(request);
}
