---
args: { links: ReceiptLink[], partitions: BalancePartition[], prior: PayableBook }
returns: ReceiptReservation[]
---
Carry existing reservations unchanged unless an authoritative release receipt exists. For each unambiguous accepted receipt link supporting a fully quantified payable portion, create a proposed reservation with exact compatible quantity, receipt/invoice/line identity and the expected shared receipt-ledger version. Reject aggregate quantities exceeding the evidenced residual receipt quantity. Reuse the prior operation ID for the same allocation; a changed allocation needs a fresh version-bound request. A proposal remains proposed until an actual atomic receipt-ledger reservation is supplied. Do not construct reservations for held, unquantified or disputed receipt links.
