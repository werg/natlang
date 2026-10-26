---
description: Sum movement receipts by exact lot and physical phase using exact units.
args: { receipts: MovementReceipt[] }
returns: Record<string, Quantity>
---
Group receipts by lot and phase, preserving their receipt IDs and source versions. Use exact decimal arithmetic; never coerce approximate strings or merge unlike units. A duplicate receipt ID is counted once and reported as duplicate evidence for later reconciliation.
