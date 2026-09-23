export default async function branches(): Promise<Branch[]> {
return await host.research.branches();
}
