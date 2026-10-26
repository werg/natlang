---
args: {"desk": "SourcingDesk", "mandate": "MandateBoundary"}
returns: "QuoteOption[]"
---
Read every quotation and amendment in the declared searched books. For each attributed supplier document call segmentCommercialTerms(document, mandate); for each proposed quote call recognizeConditionalOffer(quote, desk.quotations). Keep indicative offers separate from binding ones. Preserve currency, quantity units, logistics/service exclusions, capacity, validity, bundle dependencies and every ambiguity. Never fill an omitted cost with zero or erase a supplier condition to make quotes comparable.
