export default async function native_search(id: string, query: string): Promise<string> {
return await host.research.nativeSearch(id, query);
}
