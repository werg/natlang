---
args: { requirements: HandoverRequirement[], manuals: AssetManualMatch[], certificates: CertificateMatch[], archive: DossierArchiveAction[], exports: string[], obligations: string[] }
returns: string[]
---
Compare every required item to the exact asset-matched manual or certificate, permitted archive evidence, accessible export task, and transferred obligation. Return the missing requirement IDs and reasons; do not mark dossier complete from document count alone.
