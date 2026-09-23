import { research } from 'natlang:services';
export default async function workspace_read(head: string, path: string): Promise<string> {
return await research.read(head, path);
}
