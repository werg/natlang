---
args: {"legs": "ExchangeLeg[]", "profiles": "MaterialProfile[]", "receipts": "TransferObservation[]", "prior": "ResidualMaterial[]"}
returns: "ResidualMaterial[]"
---
For each actual transfer observation call verifyPartialHandoff(receipt, legs, profiles). Update quantity only once for matching leg and receipt identities. Keep accepted, rejected, reserved and available quantities in one exact unit basis with conservation against the offered lot. Release the unaccepted remainder only when actual custody and reservation state permit it. Preserve rejection reasons as future matching evidence without generalizing one unit's condition to every lot.
