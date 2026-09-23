import { research } from 'natlang:services';
export default async function affected(head: string, changed: string[]): Promise<string[]> {
return await research.affected(head, changed);
}
