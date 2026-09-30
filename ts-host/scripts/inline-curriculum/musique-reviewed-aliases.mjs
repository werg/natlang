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
