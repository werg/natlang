import { research } from 'natlang:services';
export default async function native_search(id: string, query: string): Promise<string> {
return await research.nativeSearch(id, query);
}
