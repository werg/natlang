export default async function audit_migration(head: string, source: string, receipt: string, spec: string): Promise<string> {
return await host.research.auditMigration(head, source, receipt, spec);
}
