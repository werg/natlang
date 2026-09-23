import type { EditorFile, EditorSnapshot, EditPatch, EditReport, CheckReport, RunReport, TraceView, ViewPanel, EditorView } from "../types.js";
import { host } from "natlang:runtime";

export default async function execute(input: string, revision: string): Promise<RunReport> {
return await host.ide.run({ input: input }, revision);
}
