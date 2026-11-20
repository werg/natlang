import { writeFileSync } from "node:fs";
import { expect, it } from "vitest";
import { allTests } from "./stages.js";
import { usesPanelRepairTeacher } from "./panel-repair-world.js";

it("extracts the system test catalog without executing the tests", () => {
  const rows = allTests().map((test) => ({
    name: test.name,
    category: test.category,
    description: test.description,
    validation: test.validation ?? "agent-goal",
    explicitOnly: Boolean(test.explicitOnly),
    requiresUnits: test.requiresUnits ?? [],
    orchestrated: Boolean(test.orchestrate),
    panelTeacher: usesPanelRepairTeacher(test),
    sendsAgentTurns: /\bsendAndWait\s*\(|\borchestrateNotebookCells\(/u.test(
      test.orchestrate?.toString() ?? "",
    ),
  }));
  expect(new Set(rows.map(({ name }) => name)).size).toBe(rows.length);
  writeFileSync(process.env.VIBESTUDIO_GOAL_CATALOG!, JSON.stringify(rows));
});
