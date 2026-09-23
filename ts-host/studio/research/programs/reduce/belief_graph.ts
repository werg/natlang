import { research } from 'natlang:services';
export default async function belief_graph(head: string): Promise<string> {
return await research.beliefGraph(head);
}
