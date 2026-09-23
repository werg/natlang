export default async function review_candidate(id: string): Promise<string> {
return await host.research.reviewCandidate(id);
}
