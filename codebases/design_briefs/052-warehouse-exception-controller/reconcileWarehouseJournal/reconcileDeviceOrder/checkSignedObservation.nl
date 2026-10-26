---
args: {"entry": "DeviceObservation", "receipts": "ScanVerification[]"}
returns: "string[]"
---
Require a verification receipt from the configured authenticated device-verification boundary matching receipt ID, device, sequence and exact payload hash. A string saying signed is not a cryptographic verification result, and the model must not invent one. Reject absent, mismatched or signature-invalid receipts. Return blocking findings; only an empty result permits journal admission. Preserve the original operator and timestamp evidence rather than adopting the sync operator as observer.
