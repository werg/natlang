export default function prepare(base: WikiPage, updates: WikiUpdate[], profile: MergeProfile): PreparedMerge {
return host.wiki.prepare(base, updates, profile);
}
