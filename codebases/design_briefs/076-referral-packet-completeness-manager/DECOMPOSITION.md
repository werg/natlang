# 076 — Referral Packet Completeness Manager

## Distinct algorithm: satisfy a versioned recipient checklist through identity-bound document references

The manager treats a referral packet as a recipient-specific checklist whose requirements point to documents, identity evidence, validity periods and signatures. It first binds the referral to an approved patient identity; a matching name is never enough to link records across charts. The receiving service's checklist version controls required items. When it changes, only pending packets with affected requirements are revalidated.

Presence is not completeness: a document can exist and still be unsigned, expired, mismatched or outside the requested episode. The manager preserves the exact gap and requests material from the responsible role. It carries an authoritative clinician priority unchanged and does not independently re-triage. A transfer moves through prepared, authorized, dispatched, recipient-acknowledged states; a missing acknowledgment triggers recipient status reconciliation before any resend of sensitive records.

The resume packet stores identity evidence refs, checklist version, episode scope, document/signature review state, clinician priority source, access disclosure ledger and delivery receipt. No packet is released from an uncertain identity or expired authorization.
