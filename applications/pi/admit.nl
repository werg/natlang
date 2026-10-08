---
description: Admission of a submission (spec §6, appendix 2 R1). Decides what a conversation does with new input or an entry write, busy or idle, and commits it with guards on the run and the inbox.
args:
  draft: SubmissionDraft
returns: AdmitResult
uses: [harness/boundary]
---
Admit draft to the conversation of the admission service. Read first, decide, then commit once; when the commit
rejects with "state changed", read again and decide again.

1. Dedupe. When draft.requestId is set: existing = admission.byRequest(draft.requestId). If there is one: when
   existing.type differs from draft.type, return { conflict: "Request <requestId> already identifies a submission of
   type <existing.type>" }; otherwise return { id: existing.id } and write nothing.
2. state = admission.state(). The conversation is busy when state.run is not null.
3. Reject. Busy, draft.type "input" and draft.whenBusy "reject": return { busy: true } and write nothing.
4. Choose the operations; expect = { run: state.run, inbox: the IDs of state.inbox in order }, plus requestAbsent:
   draft.requestId when it is set.
   a. Queue path, when busy, or idle with a non-empty state.inbox: ops start with { op: "queue", draft }.
      - Busy: that is all.
      - Idle: the queued items are older than the new one and go first. Call boundary(items, state.steeringMode,
        state.followUpMode, "final", state.activeStart) with items = state.inbox followed by the new item
        { id: 0, mode, content or entry } (mode "write" for a write; "steer" when draft.whenBusy is "steer"; otherwise
        "followUp"; content is draft.content, entry is draft.entry). Add { op: "boundary", selection } with its result.
   b. Direct path, idle with an empty inbox:
      - a write is stale when draft.entry.head is a number, state.activeStart is not null, and draft.entry.head is
        less than state.activeStart: ops = [{ op: "stale", draft }]; otherwise ops = [{ op: "write", draft }]. A write
        never starts a run;
      - an input: ops = [{ op: "input", draft }].
5. result = admission.commit(ops, expect). Return { id: result.id }.
