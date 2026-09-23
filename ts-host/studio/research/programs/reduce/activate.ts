import { research } from 'natlang:services';
export default async function activate(head: string, candidate: string): Promise<CommitResult> {
return await research.activate(head, candidate);
}
