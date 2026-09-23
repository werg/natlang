import { research } from 'natlang:services';
export default async function diff(left: string, right: string): Promise<string> {
return await research.diff(left, right);
}
