export default function validate(lock: Lock): InstallCheck {
try {
  const bundle = host.packages.bundle(lock);
  return { ok: true, revision: bundle.revision, detail: '' };
} catch (error) {
  return { ok: false, revision: '', detail: String(error) };
}
}
