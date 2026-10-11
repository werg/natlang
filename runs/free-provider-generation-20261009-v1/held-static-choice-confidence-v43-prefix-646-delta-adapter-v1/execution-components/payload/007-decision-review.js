import { canonical } from '../adaptation/identity.js';
import { hexDigest } from './hash.js';
export function nativeDecisionTargetDigest(target) {
    return hexDigest(canonical(JSON.parse(JSON.stringify(target))));
}
export function validNativeDecisionApproval(approval, expected) {
    if (!approval || typeof approval !== 'object' || Array.isArray(approval))
        return false;
    const value = approval;
    return value.schema === 'natlang.native-decision-approval/1' &&
        value.trajectory_id === expected.trajectory_id && value.source_row_sha256 === expected.source_row_sha256 &&
        value.decision_index === expected.decision_index && value.target_sha256 === expected.target_sha256 &&
        [value.source_row_sha256, value.target_sha256, value.review_sha256].every(hash => typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)) &&
        Number.isSafeInteger(value.decision_index) && Number(value.decision_index) >= 0 &&
        typeof value.reason === 'string' && !!value.reason.trim() && Array.isArray(value.evidence) &&
        value.evidence.length > 0 && value.evidence.every(item => typeof item === 'string' && !!item.trim());
}
