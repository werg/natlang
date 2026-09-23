import { research } from 'natlang:services';
export default async function audit_migration(head: string, source: string, receipt: string, spec: string): Promise<string> {
return await research.auditMigration(head, source, receipt, spec);
}
