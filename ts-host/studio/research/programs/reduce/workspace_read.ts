export default async function workspace_read(head: string, path: string): Promise<string> {
return await host.research.read(head, path);
}
