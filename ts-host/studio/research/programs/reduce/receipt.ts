import { research } from 'natlang:services';
export default async function receipt(id: string): Promise<string> {
return await research.receipt(id);
}
