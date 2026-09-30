/** Exact, evidence-hash-scoped MuSiQue oracle aliases reviewed 2026-09-30. */
import { createHash } from 'node:crypto';

const sha256 = text => createHash('sha256').update(text).digest('hex');

const SOURCE_SNAPSHOT_SHA256 = 'cb4c55dfe2de2daa36a567771e862d8575ce4fa64e68631d7055b09351de28e5';

const REVIEWED_ALIASES = Object.freeze({
  '4hop1__205937_144938_83779_69861': Object.freeze({
    primary: 'the North Korean - Chinese forces',
    accepted: 'North Korea and China',
    rationale: 'The supplied Korean War evidence identifies the Soviet-supported group as the North Korean-Chinese forces; the supplied South Korea article says the Soviet Union and China backed North Korea. The answer names those same parties.',
    evidence: Object.freeze([
      Object.freeze({ path: 'articles/14.md', sha256: '0e28948f6bb38d4bc51f89c44d44e9066907fe009556cff7cddeef2ba89e2b24', text: 'to aid the North Korean - Chinese forces against the United Nations Forces.' }),
      Object.freeze({ path: 'articles/0.md', sha256: '5b699e06412494ddeffd1bafacb9c7faeb6893dec4c237e9e4d2efad880f4102', text: 'The Soviet Union and China backed North Korea' }),
    ]),
  }),
  '3hop1__478606_751065_78953': Object.freeze({
    primary: 'Lying off the north - western coast of the European mainland',
    accepted: 'Western Europe',
    rationale: 'The supplied United Kingdom article directly identifies the UK as a sovereign country in western Europe and states the more specific coast description used by the primary answer. Western Europe is a correct, less specific answer to the location question.',
    evidence: Object.freeze([
      Object.freeze({ path: 'articles/8.md', sha256: '49169269d2bd6fe4152973cdcc3de324214c3c9d33fb6e598f08ef0e41930623', text: 'is a sovereign country in western Europe. Lying off the north - western coast of the European mainland' }),
    ]),
  }),
  '2hop__128979_90736': Object.freeze({
    primary: '1985',
    accepted: 'June 19, 1985',
    rationale: 'The supplied WAJM article places the station in Atlantic City; the supplied Golden Nugget Atlantic City article identifies Atlantic City as its location and explicitly gives its opening date as June 19, 1985. That date is a more precise supported answer consistent with the primary year answer.',
    evidence: Object.freeze([
      Object.freeze({ path: 'articles/8.md', sha256: '27b061349f5857e375e7a1bd63f5b60bde3fca6a448ea2b502f527f93ee2e7cd', text: 'WAJM, assigned to 88.9 FM and licensed to Atlantic City, New Jersey' }),
      Object.freeze({ path: 'articles/10.md', sha256: '924086dc3cc92d11056f890bbe326c959130e395da514d2251ec0124640612a6', text: 'Golden Nugget Atlantic City Location Atlantic City, New Jersey Address 1 Castle Boulevard Opening date June 19, 1985' }),
    ]),
  }),
});

// These exact source aliases were previously admitted but do not answer the
// question at the required level. Preserve the original gold and source files;
// future IRs should remove only these pinned alternates.
const REVIEWED_ALIAS_REMOVALS = Object.freeze({
  '2hop__136323_160978': Object.freeze({
    baseIrId: 'inline-curriculum:source_musique:edc16cf1494f27a7741d:v1',
    primary: 'Islamic mathematics',
    rejected: 'Islam',
    prompt: '---\nargs: {}\nreturns: "string"\nkind: directory-reducer\n---\nUnder the Arab Empire, what subject was studied in the city that Abu Hassan was in?\nRead the articles to answer. Return only the answer text. Preserve the workspace.\n',
    rationale: 'The question asks for the subject studied; the supplied paragraph names Islamic mathematics. Islam alone is broader and can denote the religion rather than that subject.',
    evidence: Object.freeze([
      Object.freeze({ path: 'articles/17.md', sha256: 'fe9a9c67320ed84b5958d6ad4d541b01460bde3eacd0f5e048f4945183753f55', text: "a favorite of the Caliph of Baghdad" }),
      Object.freeze({ path: 'articles/10.md', sha256: '18317147ddc19d5129e3fa91dbbe2c99d443215a2b045709887d35b2298f8feb', text: 'an important center of study for Islamic mathematics' }),
    ]),
  }),
  '2hop__130422_69489': Object.freeze({
    baseIrId: 'inline-curriculum:source_musique:e403cfa783ea6d1932b0:v1',
    primary: '21 or older.',
    rejected: 'Gun laws in Iowa',
    prompt: '---\nargs: {}\nreturns: "string"\nkind: directory-reducer\n---\nWhat is the minimum age requirement to buy a handgun in the state where Wartburg College is located?\nRead the articles to answer. Return only the answer text. Preserve the workspace.\n',
    rationale: 'The question asks for a minimum age. The title of the evidence article is not an answer to that question; the body states qualified applicants must be aged 21 or older.',
    evidence: Object.freeze([
      Object.freeze({ path: 'articles/0.md', sha256: 'ffef5cea04be0accc780bc4e6375594d840249eea982748fbc3736a5cfb34e09', text: 'located in Waverly, Iowa' }),
      Object.freeze({ path: 'articles/9.md', sha256: 'b0b07c0d37d4d3d700d8f1eaea4605a96b18cfa61c52bf5720f5b2b70148194e', text: 'shall be issued to qualified applicants aged 21 or older' }),
    ]),
  }),
});

/** Apply only an exact, snapshot-scoped removal of a previously accepted alias. */
export function reviewedMusiqueAliasRemovalFor({ sourceId, irId, snapshotSha256, primary, prompt, oracle, files }) {
  const entry = REVIEWED_ALIAS_REMOVALS[sourceId];
  if (!entry) return null;
  const oracleKeys = oracle && typeof oracle === 'object' && !Array.isArray(oracle) ? Object.keys(oracle).sort() : [];
  const exactOracleShape = oracleKeys.length === 2 && oracleKeys[0] === 'alternates' && oracleKeys[1] === 'level' &&
    oracle.level === 'normalized' && Array.isArray(oracle.alternates);
  const oracleState = exactOracleShape && oracle.alternates.length === 1 && oracle.alternates[0] === entry.rejected ? 'base' :
    exactOracleShape && oracle.alternates.length === 0 ? 'removed' : null;
  const expectedIrId = oracleState === 'removed' ? `${entry.baseIrId}:reviewed-oracle-alias-removal-v2` : entry.baseIrId;
  if (snapshotSha256 !== SOURCE_SNAPSHOT_SHA256 || primary !== entry.primary || prompt !== entry.prompt ||
      !oracleState || irId !== expectedIrId || !files ||
      !entry.evidence.every(({ path, sha256: expectedHash, text }) => {
        const content = files[path];
        return typeof content === 'string' && sha256(content) === expectedHash && content.includes(text);
      })) {
    throw new Error(`reviewed_musique_alias_removal_evidence_mismatch:${sourceId}`);
  }
  return {
    rejected: entry.rejected,
    audit: {
      registry: 'musique-reviewed-alias-removals/2026-09-30-v2',
      review: 'root-reviewed-invalid-answer-alias-removal',
      source_id: sourceId,
      base_ir_id: entry.baseIrId,
      source_snapshot_sha256: SOURCE_SNAPSHOT_SHA256,
      primary: entry.primary,
      rejected: entry.rejected,
      task_prompt: entry.prompt,
      original_oracle: { level: 'normalized', alternates: [entry.rejected] },
      rationale: entry.rationale,
      evidence: entry.evidence.map(({ path, sha256, text }) => ({ path, sha256, text })),
    },
    oracleState,
  };
}

/**
 * Apply only reviewed answer aliases whose source snapshot, exact primary gold, and
 * full visible evidence files still match the reviewed record. Known-source drift
 * is an error so future builds cannot silently drop or misapply a review.
 */
export function reviewedMusiqueAliasFor({ sourceId, snapshotSha256, primary, files }) {
  const entry = REVIEWED_ALIASES[sourceId];
  if (!entry) return null;
  if (snapshotSha256 !== SOURCE_SNAPSHOT_SHA256 || primary !== entry.primary || !files ||
      !entry.evidence.every(({ path, sha256: expectedHash, text }) => {
        const content = files[path];
        return typeof content === 'string' && sha256(content) === expectedHash && content.includes(text);
      })) {
    throw new Error(`reviewed_musique_alias_evidence_mismatch:${sourceId}`);
  }
  return {
    accepted: entry.accepted,
    audit: {
      registry: 'musique-reviewed-aliases/2026-09-30-v1',
      review: 'root-reviewed-semantic-alias',
      source_id: sourceId,
      source_snapshot_sha256: SOURCE_SNAPSHOT_SHA256,
      primary: entry.primary,
      accepted: entry.accepted,
      rationale: entry.rationale,
      evidence: entry.evidence.map(({ path, sha256, text }) => ({ path, sha256, text })),
    },
  };
}
