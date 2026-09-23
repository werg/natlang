export default function report(revision: string, checks: Validation): MigrationReport {
return host.repository.report(revision, checks);
}
