export default async function activate(head: string, candidate: string): Promise<CommitResult> {
return await host.research.activate(head, candidate);
}
