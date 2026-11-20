#!/usr/bin/env node
import { execFileSync } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { parseArgs } from "node:util";

const { values } = parseArgs({
  options: {
    host: { type: "string" }, templates: { type: "string" },
    baseline: { type: "string" }, out: { type: "string" },
    workspace: { type: "string", multiple: true }, model: { type: "string" },
    plan: { type: "boolean" },
  },
});
for (const flag of ["host", "templates", "baseline", "out"]) {
  if (!values[flag]) throw new Error(`Pass --${flag}`);
}
const output = path.resolve(values.out);
if (existsSync(output)) throw new Error(`The report path already exists: ${output}`);
const host = path.resolve(values.host);
const templates = path.resolve(values.templates);
const baselineRows = readFileSync(values.baseline, "utf8").trimEnd().split("\n").map((line) => JSON.parse(line));
const baseline = baselineRows[0];
if (baseline?.schema !== "vibestudio.open-goal-inventory/1") throw new Error("Pass a pinned open-goal inventory as the baseline");
const baselineSkills = new Set(baselineRows.filter(({ kind }) => kind === "skill").map(({ id }) => id));
const git = (root, ...args) => execFileSync("git", ["-C", root, ...args], { encoding: "utf8", maxBuffer: 16 * 1024 * 1024 });
const head = (root) => git(root, "rev-parse", "HEAD").trim();
const sha256 = (content) => createHash("sha256").update(content).digest("hex");
const sourceKey = (url) => url.replace(/^git\+/u, "").replace(/\.git$/u, "").toLowerCase();
const hostPin = head(host);
const { templates: registry } = JSON.parse(git(host, "show", `${hostPin}:templates/registry.json`));
const checkout = (id) => path.join(templates, id);
const roots = { host, ...Object.fromEntries(registry.map(({ id }) => [id, checkout(id)])) };
const pins = Object.fromEntries(Object.entries(roots).map(([id, root]) => [id, head(root)]));
const instantiable = registry.filter(({ id, role }) =>
  role === "personal" || role === "system" || (role === "catalog" && registry.some((row) =>
    row.role === "development" && row.consumers?.includes(id)
  ))).map(({ id }) => id);
const selected = values.workspace ?? instantiable;
if (new Set(selected).size !== selected.length || selected.some((id) => !instantiable.includes(id))) {
  throw new Error(`Select distinct instantiable workspaces: ${instantiable.join(", ")}`);
}
function checkPins() {
  for (const [id, root] of Object.entries(roots)) {
    if (head(root) !== pins[id] || git(root, "status", "--porcelain=v1").trim())
      throw new Error(`Use an unchanged clean source checkout for ${id}`);
  }
}
if (!values.plan) checkPins();

const hostImport = (file) => import(pathToFileURL(path.join(host, file)).href);
const { parseSkillFrontmatter } = await hostImport("src/server/vcsHost/workspaceSkills.ts");
const { splitRepoPath, normalizeWorkspaceRepoPath } = await hostImport("packages/shared/src/runtime/entitySpec.ts");
const { templateRepositories, parseTemplateManifestContent } = await hostImport("packages/workspace/src/templateManifest.ts");
const { templateRepositoryOwners } = await hostImport("packages/workspace/src/templateManifestMerge.ts");
const { WORKSPACE_SYSTEM_EPOCH } = await hostImport("packages/shared/src/vcs/systemEpoch.ts");
// Workspace skill and template parsing are owned by the installed host; do not port the rules here.
function eligibleRepo(skillPath) {
  const split = splitRepoPath(skillPath);
  if (split?.repoRelPath !== "SKILL.md") return null;
  try { return normalizeWorkspaceRepoPath(split.repoPath); } catch { return null; }
}
const sourceByUrl = new Map();
const sourceDocs = [];
for (const row of registry) {
  const { id, url } = row;
  const root = checkout(id);
  const pin = pins[id];
  const files = git(root, "ls-tree", "-r", "--name-only", "-z", pin).split("\0").filter(Boolean);
  const manifest = parseTemplateManifestContent(git(root, "show", `${pin}:meta/vibestudio.yml`), WORKSPACE_SYSTEM_EPOCH);
  // Managed development injects its harness as a dependency of every registered consumer.
  for (const layer of registry) {
    if (layer.role === "development" && layer.consumers?.includes(id) &&
        !manifest.dependencies.some((dep) => sourceKey(dep.url) === sourceKey(layer.url))) {
      manifest.dependencies.push({ url: layer.url });
    }
  }
  const layer = { id, label: url, manifest, repositories: templateRepositories(files) };
  sourceByUrl.set(sourceKey(url), layer);
  for (const skillPath of files.filter((f) => f === "SKILL.md" || f.endsWith("/SKILL.md"))) {
    const content = git(root, "show", `${pin}:${skillPath}`);
    const front = parseSkillFrontmatter(content);
    const repoPath = eligibleRepo(skillPath);
    const sourceId = `${id}:${skillPath}`;
    sourceDocs.push({
      id: sourceId, template: id, skillPath, repoPath, sha256: sha256(content),
      agentVisible: front.agentVisible !== false, inBaseline: baselineSkills.has(sourceId),
      ...(repoPath ? { name: front.name ?? path.posix.basename(repoPath), description: front.description ?? "" } : {}),
    });
  }
}
function planWorkspace(workspace) {
  const layers = [];
  const visiting = new Set();
  function visit(url) {
    const key = sourceKey(url);
    const layer = sourceByUrl.get(key);
    if (!layer) throw new Error(`No pinned template for ${url}`);
    if (visiting.has(key)) throw new Error(`Template dependency cycle at ${url}`);
    if (layers.includes(layer)) return;
    visiting.add(key);
    for (const dep of layer.manifest.dependencies) visit(dep.url);
    visiting.delete(key);
    layers.push(layer);
  }
  visit(registry.find(({ id }) => id === workspace).url);
  const owners = templateRepositoryOwners(layers);
  const sources = layers.map(({ id }) => id);
  const expected = sourceDocs.filter((doc) => sources.includes(doc.template)).map((doc) => {
    const owner = doc.repoPath ? owners.get(doc.repoPath)?.id ?? null : null;
    const disposition = !doc.repoPath ? "outside-discovery-scope" :
      owner !== doc.template ? "shadowed-repository" : doc.agentVisible ? "visible" : "agent-hidden";
    return { source: doc.id, path: doc.skillPath, owner, disposition };
  });
  return { workspace, sources, expected };
}

const env = { ...process.env, VIBESTUDIO_TEMPLATE_CHECKOUTS: templates, VIBESTUDIO_MAX_CONCURRENT_BUILDS: "1" };
const cli = (tool, ...args) => execFileSync("pnpm", ["--silent", tool, ...args], {
  cwd: host, env, encoding: "utf8", maxBuffer: 16 * 1024 * 1024,
});
function observe(workspace) {
  const instance = `skill-inventory-${workspace}-${randomUUID().slice(0, 8)}`;
  const session = "skill-inventory";
  let failure;
  try {
    const doctor = JSON.parse(cli("system-test", "--instance", instance, "--workspace", workspace,
      "doctor", "--approve-startup", ...(values.model ? ["--model", values.model] : [])));
    const server = doctor.checks?.find(({ name }) => name === "server")?.data;
    if (!doctor.ok || !server?.workspaceId) throw new Error("The managed workspace did not supply an exact identity");
    const identity = JSON.parse(cli("cli", "--instance", instance, "agent", "attach", session, "--json"));
    if (identity.workspaceId !== server.workspaceId || !identity.contextId)
      throw new Error("The skill probe agent session belongs to another workspace");
    const skillCli = (...args) => JSON.parse(cli("cli", "--instance", instance, "agent",
      "skills", ...args, "--session", session, "--json"));
    const list = skillCli();
    if (!Array.isArray(list) || new Set(list.map((row) => row?.skillPath)).size !== list.length)
      throw new Error("Workspace.listSkills returned malformed or repeated entries");
    const skills = list.map(({ name, description, dirPath, skillPath }) => {
      if (typeof name !== "string" || typeof description !== "string" || eligibleRepo(skillPath) !== dirPath)
        throw new Error("Workspace skill path did not match its declared repository");
      const content = skillCli(dirPath);
      if (typeof content !== "string") throw new Error("Workspace.readSkill returned no source text");
      return { name, description, dirPath, skillPath, sha256: sha256(content), agentVisible: true };
    });
    return { workspaceId: identity.workspaceId, contextId: identity.contextId,
      ...(identity.entityId ? { entityId: identity.entityId } : {}), instance, skills };
  } catch (error) {
    failure = error;
    throw error;
  } finally {
    try {
      cli("system-test", "--instance", instance, "--workspace", workspace, "stop");
    } catch (cleanupError) {
      const message = `The managed skill probe instance ${instance} failed to retire`;
      if (failure) throw new AggregateError([failure, cleanupError], message);
      throw new Error(message, { cause: cleanupError });
    }
  }
}
function compare(plan, observation) {
  const tracks = new Map(sourceDocs.map((doc) => [doc.id, doc]));
  const expected = new Map(plan.expected.filter((row) => row.disposition === "visible").map((row) => [row.path, row.source]));
  const observed = new Set(observation.skills.map(({ skillPath }) => skillPath));
  const missing = [...expected.keys()].filter((skillPath) => !observed.has(skillPath));
  const mismatches = [];
  const skills = observation.skills.map((row) => {
    const source = expected.get(row.skillPath) ?? null;
    const doc = tracks.get(source);
    const candidates = sourceDocs.filter((candidate) => plan.sources.includes(candidate.template) &&
      candidate.skillPath === row.skillPath && candidate.sha256 === row.sha256).map(({ id }) => id);
    if (!doc || doc.sha256 !== row.sha256 || doc.name !== row.name || doc.description !== row.description) {
      mismatches.push({ path: row.skillPath, expectedSource: source, digestMatchingSources: candidates });
    }
    return { ...row, expectedSource: source, digestMatchingSources: candidates };
  });
  return { ...plan, observed: { ...observation, skills }, missing, mismatches,
    matchesPinnedSources: missing.length === 0 && mismatches.length === 0 };
}

function failureText(error) {
  if (error instanceof AggregateError) return [error.message, ...error.errors.map(failureText)].join(" | ");
  if (error instanceof Error && error.cause) return error.message + " | " + failureText(error.cause);
  return String(error?.message ?? error);
}

const workspaces = [];
for (const workspace of selected) {
  const plan = planWorkspace(workspace);
  if (values.plan) workspaces.push(plan);
  else {
    try {
      checkPins();
      const observed = compare(plan, observe(workspace));
      checkPins();
      workspaces.push(observed);
    } catch (error) {
      workspaces.push({ ...plan, error: failureText(error) });
      break;
    }
  }
}
const report = {
  schema: "vibestudio.installed-skill-inventory/1",
  mechanism: values.plan ? "pinned-source-plan" : "workspace.listSkills+workspace.readSkill",
  pins, baselinePins: baseline.pins,
  compositionLayers: registry.filter(({ role }) => role === "base" || role === "development").map(({ id }) => id),
  instantiableWorkspaces: instantiable, workspaces, sourceDocs,
  baselineSourcesMissingFromPins: [...baselineSkills].filter((id) => !sourceDocs.some((doc) => doc.id === id)),
  complete: !values.plan && workspaces.length === instantiable.length &&
    workspaces.every((row) => row.matchesPinnedSources),
};
writeFileSync(output, JSON.stringify(report, null, 2) + "\n", { flag: "wx", mode: 0o600 });
for (const row of workspaces) {
  const count = row.observed?.skills.length ?? row.expected.filter(({ disposition }) => disposition === "visible").length;
  console.log(`${row.workspace}  ${count}  ${row.error ?? (values.plan ? "planned" : row.matchesPinnedSources ? "verified" : "mismatch")}`);
}
if (!values.plan && workspaces.some((row) => !row.matchesPinnedSources)) process.exitCode = 1;
