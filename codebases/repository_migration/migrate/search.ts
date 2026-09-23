export default function search(query: string, revision: string): SearchResult {
return host.repository.search(query, revision);
}
