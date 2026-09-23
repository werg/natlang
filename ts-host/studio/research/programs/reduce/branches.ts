import { research } from 'natlang:services';
export default async function branches(): Promise<Branch[]> {
return await research.branches();
}
