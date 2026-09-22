---
args:
  request: PackageRequest
  locks: Lock[]
returns: string
---
Choose the ID of one offered lock that satisfies the user's purpose. Prefer
the newest compatible root version unless the purpose gives a reason to retain
an older compatible version. Do not change any package pin or invent a lock.
Write exactly the chosen lock ID.
