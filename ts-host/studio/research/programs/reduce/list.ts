import { research } from 'natlang:services';
export default async function list(head: string): Promise<Entry[]> {
return await research.list(head);
}
