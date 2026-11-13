---
description: Classify a customer message as billing, technical or other.
args:
  message: string
returns: '"billing" | "technical" | "other"'
readout: decision
---
Classify message as billing (charges, invoices or payments), technical (errors or bugs), or other. Return only the matching label.
