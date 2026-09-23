import type { Request, File, Clip, Plan, Receipt, Inspection, Assessment, MediaResult } from "../types.js";
import { host } from "natlang:runtime";

export default async function probe(input: string): Promise<Clip> {
return await host.media.probe(input);
}
