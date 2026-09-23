export default async function search(head: string, query: string): Promise<Hit[]> {
return await host.research.search(head, query);
}
