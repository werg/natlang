import { research } from 'natlang:services';
export default async function search(head: string, query: string): Promise<Hit[]> {
return await research.search(head, query);
}
