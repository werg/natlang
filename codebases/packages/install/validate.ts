/*---
engine: typescript-host
args:
  lock: Lock
returns: InstallCheck
---*/
try {
  const bundle = host.packages.bundle(args.lock);
  return { ok: true, revision: bundle.revision, detail: '' };
} catch (error) {
  return { ok: false, revision: '', detail: String(error) };
}
