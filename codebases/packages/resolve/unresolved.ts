export default function unresolved(request: PackageRequest): Resolution {
return { status: 'unresolved', lock: { id: '', root: request.name,
  engine: request.engine, packages: [] }, alternatives: 0,
  detail: 'No compatible lock satisfies the declared constraints and engine.' };
}
