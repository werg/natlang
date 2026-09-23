export default async function affected(head: string, changed: string[]): Promise<string[]> {
return await host.research.affected(head, changed);
}
