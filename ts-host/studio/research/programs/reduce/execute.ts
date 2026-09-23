import { research } from 'natlang:services';
export default async function execute(head: string, root: string, inputs: string, call_id: string): Promise<Receipt> {
return await research.execute(head, root, inputs, call_id);
}
