import { research } from 'natlang:services';
export default async function review_reconciliation(candidates: string[]): Promise<string> {
return await research.reviewReconciliation(candidates);
}
