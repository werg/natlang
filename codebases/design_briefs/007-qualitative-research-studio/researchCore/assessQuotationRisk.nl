---
args: { passages: InterviewPassage[], consentEvents: EvidencePage[] }
returns: QuotationRiskReview
---
Inspect direct and indirect identification cues in each candidate quotation against its consent scope and audience. Preserve the original in restricted analysis where allowed; for export, propose masking or paraphrase and retain a reviewer question when re-identification risk remains. Do not expose participant identity to unauthorized reviewers.
