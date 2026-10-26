---
args: {"state": "ClaimReleaseState", "receipts": "PublicationReceipt[]"}
returns: "PublicationReadback"
---
For each receipt find the exact bundle, page, candidate bytes, base hash and approved operation. Contradictory or mismatched evidence leaves the page in repairQueue. An actual matching published receipt records that page once. A failed receipt records the specific repair; an unknown outcome requires readback before resend. Preserve successful pages when another page fails. FullyPublished is true only when every intended page has matching successful receipt and no unresolved repair or source invalidation; one success never closes the bundle.
