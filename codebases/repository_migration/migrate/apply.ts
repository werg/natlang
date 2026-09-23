export default function apply(revision: string, patch: Patch[]): RepoSnapshot {
return host.repository.apply(revision, patch);
}
