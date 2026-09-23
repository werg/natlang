import { research } from 'natlang:services';
export default async function review_candidate(id: string): Promise<string> {
return await research.reviewCandidate(id);
}
