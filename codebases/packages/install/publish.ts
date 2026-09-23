export default async function publish(lock: Lock, target: string): Promise<InstallReport> {
try { return await host.packages.install(lock, target); }
catch (error) { return { status: 'failed', target: target,
  revision: '', detail: String(error) }; }
}
