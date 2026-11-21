import { cpSync, lstatSync, mkdtempSync, readdirSync, realpathSync, renameSync, rmSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

async function inspect(directory) {
  const api = await import(pathToFileURL(path.join(directory, "scripts/infrastructure-package-cache.mjs")));
  return { api, plan: api.inspectInfrastructurePackageBuilds({ cwd: directory }) };
}

function requireRegularTree(directory) {
  if (!lstatSync(directory).isDirectory()) throw new Error(`Expected a regular directory at ${directory}`);
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const child = path.join(directory, entry.name);
    if (entry.isDirectory()) requireRegularTree(child);
    else if (!entry.isFile()) throw new Error(`Expected a regular infrastructure output at ${child}`);
  }
}

if (process.argv.length !== 4) throw new Error("Specify prepared and staging Host checkouts");
const [source, release] = process.argv.slice(2).map((directory) => realpathSync(directory));
if (source === release) throw new Error("Specify separate prepared and staging Host checkouts");
const prepared = await inspect(source);
const staging = await inspect(release);
const available = new Map(prepared.plan.states.map((state) => [state.name, state]));
if (!available.size || prepared.plan.dirty.length || available.size !== staging.plan.states.length) {
  throw new Error("The prepared Host needs a complete, verified infrastructure build");
}

for (const state of staging.plan.states) {
  const original = available.get(state.name);
  if (original?.inputDigest !== state.inputDigest || original.relativeDirectory !== state.relativeDirectory) {
    throw new Error(`Prepared infrastructure inputs differ for ${state.name}`);
  }
  requireRegularTree(path.join(original.directory, "dist"));
}

const missing = [];
for (const state of staging.plan.states) {
  const destination = path.join(state.directory, "dist");
  if (!lstatSync(destination, { throwIfNoEntry: false })) {
    missing.push(state);
    continue;
  }
  requireRegularTree(destination);
  if (JSON.stringify(state.outputs) !== JSON.stringify(available.get(state.name).outputs)) {
    throw new Error(`Cannot seed ${state.name} over existing build outputs at ${destination}`);
  }
}
for (const state of missing) {
  const temporary = mkdtempSync(path.join(state.directory, ".stage-infrastructure-"));
  try {
    const copied = path.join(temporary, "dist");
    cpSync(path.join(available.get(state.name).directory, "dist"), copied, { recursive: true });
    renameSync(copied, path.join(state.directory, "dist"));
  } finally {
    rmSync(temporary, { recursive: true, force: true });
  }
}

const copied = staging.api.inspectInfrastructurePackageBuilds({ cwd: release });
for (const state of copied.states) {
  if (JSON.stringify(state.outputs) !== JSON.stringify(available.get(state.name).outputs)) {
    throw new Error(`Copied infrastructure output hashes differ for ${state.name}`);
  }
}
staging.api.writeInfrastructurePackageCache(copied);
console.log(`Verified matching infrastructure inputs and output hashes for ${copied.states.length} packages`);
