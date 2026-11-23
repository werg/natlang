# Inbox Pilot (`applications/inbox-pilot/`): Decomposition

Tier 2 flagship A4 (`plans/semantic-fp/06_FLAGSHIP_APPLICATIONS.md` §3): triages daily mail against standing prose
rules with crisp header fast-paths, fenced `Untrusted<string>` boundaries, asymmetric escalation on risky messages,
and `ONCE_EFFECTS` idempotent side effects.

## Policy vs Mechanism

- **Exact mechanism (TypeScript, `pilot.ts`):**
  - `isRoutineNewsletterOrReceipt`: exact transport-header check (`List-Unsubscribe`, `List-Id` + `Precedence`,
    `X-Receipt-Id` + `Auto-Submitted`, `X-Mail-Category`), the crisp side of the `pluggable` routine-mail fast path
    (`default: 'crisp'`; its nl side is `decideAction.nl` answering `archive`; `shadow` runs both and serves crisp)
    so routine newsletters and receipts archive with zero model calls. Headers are sender-controlled: a forged
    bulk header can only archive a message (non-destructive), never draft or act on it.
  - `fenceMessage`: wraps every incoming `subject` and `body` with `untrusted(...)` before any `.nl` call so embedded
    prompt injections remain quoted data rather than instructions.
  - `compileStandingRules`: compiles standing prose rules that supply labeled examples via `compileRule`.
  - `createOnceMailService`: declares `[ONCE_EFFECTS]: ['archive', 'saveDraft', 'hold']` and enforces at-most-one
    mutating mail effect per `messageId`, concurrent duplicate deliveries included.
  - `processInbox`: fans out over incoming messages concurrently via `semanticMap`, gating `decideAction.nl` with
    asymmetric `escalate(decideAction, { to: 'teacher', floor: 0.85, risk: { escalate: 0.25 } })` and translating
    `'escalate'` decisions into safe non-destructive holds.
- **Typed semantic policy (`.nl` leaves):**
  - `decideAction.nl` (`readout: decision`, returns `InboxAction`): classifies non-routine untrusted messages into
    `'archive' | 'draft-reply' | 'hold' | 'escalate'` under the standing rules passed as its `rules` argument.
  - `draftReply.nl` (returns `ReplyDraft`, an `Is<string, ...>` refinement): writes a concise, polite reply that
    follows standing rule guidance without making unapproved commitments.
