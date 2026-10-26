---
args: {"receipt": "EvidencePage"}
returns: "ReceiptComponent[]"
---
Read only visible digits and explicit currency/scale. Convert exact decimal text into integer minor units, preserving signs and original dates. Check component totals against the receipt total using integer arithmetic, accounting for stated discounts and included taxes exactly once. A tax or tip printed separately stays separate. Return amount null and the missing field for an illegible value; a mismatch cannot be repaired by inventing a balancing fee.
