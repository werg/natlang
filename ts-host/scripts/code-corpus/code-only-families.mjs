// Deliberately outside the portable native replay boundary. These examples are
// code proposals only: cases stay empty and evidence never claims replay.
export const CODE_ONLY_GENERATOR = 'natlang.code_only_curriculum/1';

export const CODE_ONLY_FAMILIES = [
  { name:'select_profile_fields', boundary:'plain_object', instruction:'Implement selectProfileFields(profile). Return a new object containing only name and email. Copy each field only when it is an own property of profile; do not mutate profile.',
    source:'function selectProfileFields(profile) {\n  const result = {};\n  for (const key of ["name", "email"]) {\n    if (Object.prototype.hasOwnProperty.call(profile, key)) result[key] = profile[key];\n  }\n  return result;\n}' },
  { name:'normalize_inventory', boundary:'array_of_objects', instruction:'Implement normalizeInventory(rows). Each row has string sku and integer quantity. Return new objects with trimmed, upper-case sku and the original quantity. Keep row order and do not mutate inputs.',
    source:'function normalizeInventory(rows) {\n  return rows.map(row => ({ sku: row.sku.trim().toUpperCase(), quantity: row.quantity }));\n}' },
  { name:'validate_service_config', boundary:'object_validation', instruction:'Implement validateServiceConfig(config). Return true only when config is a non-null object with an integer port from 1 through 65535 and mode exactly "development" or "production". Return false for every other value.',
    source:'function validateServiceConfig(config) {\n  return config !== null && typeof config === "object" && Number.isInteger(config.port) && config.port >= 1 && config.port <= 65535 && (config.mode === "development" || config.mode === "production");\n}' },
  { name:'parse_positive_integer', boundary:'explicit_error_result', instruction:'Implement parsePositiveInteger(text). Accept only a nonempty sequence of ASCII digits representing a safe integer greater than zero. Return { ok: true, value: number } when valid and { ok: false, error: "invalid positive integer" } otherwise. Do not throw.',
    source:'function parsePositiveInteger(text) {\n  if (typeof text !== "string" || !/^[0-9]+$/.test(text)) return { ok: false, error: "invalid positive integer" };\n  const value = Number(text);\n  if (!Number.isSafeInteger(value) || value <= 0) return { ok: false, error: "invalid positive integer" };\n  return { ok: true, value };\n}' },
  { name:'load_dashboard', boundary:'async_composition', instruction:'Implement async loadDashboard(userId, api). api has async getUser(id) and getNotifications(id) methods. Start both calls before awaiting either, then return { user, notifications }. Let either rejection propagate unchanged.',
    source:'async function loadDashboard(userId, api) {\n  const [user, notifications] = await Promise.all([api.getUser(userId), api.getNotifications(userId)]);\n  return { user, notifications };\n}' },
  { name:'with_error_context', boundary:'async_error_handling', instruction:'Implement async withErrorContext(operation). Await operation(). If it succeeds, return its value. If it rejects with an Error, throw a new Error whose message is "operation failed: " followed by the original message and whose cause is the original error. For non-Error rejections, use the text "unknown error".',
    source:'async function withErrorContext(operation) {\n  try {\n    return await operation();\n  } catch (error) {\n    const detail = error instanceof Error ? error.message : "unknown error";\n    throw new Error(`operation failed: ${detail}`, { cause: error });\n  }\n}' },
  { name:'sha256_hex', boundary:'node_builtin_dependency', instruction:'Implement sha256Hex(text) in Node.js using the built-in node:crypto module. Hash the UTF-8 bytes of text with SHA-256 and return the lowercase hexadecimal digest. Do not install dependencies.',
    source:'import { createHash } from "node:crypto";\nfunction sha256Hex(text) {\n  return createHash("sha256").update(text, "utf8").digest("hex");\n}' },
];

export function syntheticCodeOnlyTasks(seed = 0) {
  if (!Number.isSafeInteger(seed)) throw new Error('seed must be a safe integer');
  return CODE_ONLY_FAMILIES.map((family, index) => {
    const group = `natlang-synthetic-code-only:${family.name}`;
    return { version:'natlang.code_task/1', id:`${CODE_ONLY_GENERATOR}:${group}:${seed}`, group_id:group,
      kind:'instruction', language:'javascript', instruction:family.instruction,
      source:{ name:'natlang-synthetic-code-only', revision:CODE_ONLY_GENERATOR, path:'generated', license:'project-generated', split:'train' },
      function:{ name:family.name, parameters:[], return_type:'unspecified', source:family.source }, cases:[],
      verification:{ status:'syntax_only_candidate', reasons:['object_async_error_and_import_boundaries_are_not_native_replay_verified'] },
      generation:{ generator:CODE_ONLY_GENERATOR, seed, index, family:family.name, boundary:family.boundary },
      behavioral_evidence:{ kind:'syntax_only', status:'not_native_replay_verified', properties:['source parses as JavaScript', 'no argument/result oracle supplied'] } };
  });
}
