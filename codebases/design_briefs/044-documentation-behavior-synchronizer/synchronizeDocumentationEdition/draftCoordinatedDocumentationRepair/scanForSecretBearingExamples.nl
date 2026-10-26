---
args: { exampleSources: Evidence[] }
returns: SecretExportFinding[]
---
Inspect code samples for literal credentials or secret-bearing values. Return only the file span and secret category needed for remediation; do not repeat the secret in any generated patch, preview, or export. Keep the bundle blocked until the owner confirms rotation through operational handling.
