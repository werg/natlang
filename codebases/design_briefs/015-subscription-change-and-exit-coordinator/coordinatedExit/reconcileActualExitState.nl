---
description: Compare actual subscription and invoice state against the exact selected exit option.
args:{request:SubscriptionExitRequest;choices:ChangeOption[];operation:SubscriptionOperation;mutations:EvidencePage[];billing:EvidencePage[]}
returns:ExitReceipt|null
---
Require a matching operation receipt and current subscription readback for every affected product. Compare next invoice, access end and data retention with the option. Represent reseller-managed or shared-owner actions still pending as incomplete; do not claim full exit from a local request. Keep pending refunds visible after closure.
