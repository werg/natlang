export default function publish(base: WikiPage, prepared: PreparedMerge, draft: MergeDraft, profile: MergeProfile): MergeReport {
return host.wiki.publish(base, prepared, draft, profile);
}
