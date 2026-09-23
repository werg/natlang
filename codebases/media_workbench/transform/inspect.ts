import type { Request, File, Clip, Plan, Receipt, Inspection, Assessment, MediaResult } from "../types.js";
import { host } from "natlang:runtime";

export default async function inspect(request: Request, plan: Plan, receipt: Receipt): Promise<Inspection> {
return await host.media.inspect(request, plan, receipt);
}
