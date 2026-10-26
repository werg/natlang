---
args: {"entries": "DeviceObservation[]", "receipts": "ScanVerification[]", "prior": "DeviceCursor[]", "history": "DeviceObservation[]"}
returns: "ScanIngress"
---
Group by device and order supplied entries by exact sequence number. For each entry call checkSignedObservation(entry, receipts), then compare its sequence and payload hash with the persisted cursor and supplied history. Identical replay does not add quantity. A reused sequence with a different payload is quarantined. Keep missing sequence ranges explicit without inventing the missing observations; advance acknowledgedThrough only across contiguous verified entries. Preserve offline entries as pending sync until the exact receipt exists. A device with no evidenced initial cursor remains pending establishment of that cursor. Return cursors, the actual admitted entries, quarantined entries and required sync requests as ScanIngress.
