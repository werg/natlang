export default function reject(base: WikiPage, prepared: PreparedMerge): MergeReport {
return { status: 'rejected', page: base, detail: prepared.detail };
}
