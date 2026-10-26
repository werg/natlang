---
args:{prior:ExitCheckpoint|null;request:SubscriptionExitRequest;services:ExitServiceSnapshot;operation:SubscriptionOperation|null;receipt:ExitReceipt|null;choices:ChangeOption[]}
returns:ExitCheckpoint
---
Bind the case to the exact account and requested subscription IDs. Preserve every consent token, mutation receipt and billing receipt as its own lineage. An identical request and receipt set is a no-op. Append only actual operation outcomes; unknown outcomes remain unknown, and partial subscription changes stay separate. Reopen consent when option, date, account version or target set changes. Track refunds after closure until an authoritative billing receipt resolves them. Advance revision for new receipt or changed consent scope only.
