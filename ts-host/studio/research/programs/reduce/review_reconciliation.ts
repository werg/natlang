export default async function review_reconciliation(candidates: string[]): Promise<string> {
return await host.research.reviewReconciliation(candidates);
}
