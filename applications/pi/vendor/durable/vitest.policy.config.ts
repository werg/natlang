// PATCH (natlang port): pi-durable's suite with the policy path selected, driven by pi's own rules over facts
// (`crispSchedulerPolicy`), and admission through the policy interface. Checks the plumbing the natural-language
// implementations use.
import { mergeConfig } from "vitest/config";
import base from "./vitest.config.ts";

export default mergeConfig(base, { test: { setupFiles: ["./test/policy-setup.ts"] } });
