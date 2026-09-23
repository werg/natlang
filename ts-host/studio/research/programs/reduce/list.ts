export default async function list(head: string): Promise<Entry[]> {
return await host.research.list(head);
}
