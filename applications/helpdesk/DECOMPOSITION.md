# Helpdesk: decomposition, part by part

Status: implemented on 2026-10-09 (the owner asked for everything to go to main; plans/OWNER_REVIEW.md lists it for review
after the fact). The sections below describe the code as built; "As built" at the end records where it differs from the
first draft.

Today the package is 194 lines of TypeScript (`index.ts`) plus 75 lines of HTTP (`server.ts`) and two small
natural-language functions (`triage.nl`, `draft_reply.nl`). The state machine is crisp, which is right. The policies
that sit in TypeScript are the response deadlines (`index.ts:44`, `153-156`), the inbox ranking (`index.ts:189-190`),
what an escalation does (`index.ts:131-136`, `server.ts:66`) and the bundling of three judgments into one triage
call (`triage.nl:5`).

Decisions:

- **fn**: its own natural-language function with a typed contract.
- **inline**: one instruction inside its caller.
- **implicit**: left to the model (rare).
- **crisp**: a TypeScript helper (plumbing, or an exact check).
- **service**: the outside world.
- **host**: mechanism around the stages (event loop, commit, durability).
- **pluggable**: one interface, a crisp and a natural-language implementation, a setting selects
  (`ts-host/src/runtime/pluggable.ts`).

The executors are small, fast models. Each function is one task a small model can finish; algorithms are numbered
steps over named data; exact arithmetic is done in eval or in the crisp commit.

## Policy

- **Natural language: judgments about conversations.** How urgent a conversation is, what it is about, what the
  customer already tried, which details are still missing, the draft reply, how long this ticket may wait under the
  desk's written service policy, what an escalation should say and to whom, and the order of the agents' queue.
- **Host: the state machine.** One `KeyedEventLoop` per ticket, `reduce` (`index.ts:99-138`), the atomic per-ticket
  store (`index.ts:49-66`), restart recovery (`index.ts:162-167`). Exactly-once by event ID, `through` counters that
  drop stale follow-ups, and timers as state (`due`, `wakeAt` at `index.ts:140`) stay crisp: they are durability and
  ordering, and a model cannot be asked to guarantee them.
- **Service: the HTTP surface and the pager.** `server.ts` validates request shape and nothing else. Paging a
  supervisor is a service call that the escalation plan requests as data.
- **Decide on a snapshot, apply a pure commit.** Each model call reads a snapshot of the conversation and returns a
  value; `reduce` applies it only when the snapshot is still current (`through === messages.length`,
  `index.ts:116`, `125`). This already follows the app rules and stays.
- **Derived values form a DAG.** Messages, then urgency, then summary and topic, then deadline, then escalation plan;
  messages, summary and missing details, then draft. Nothing reads a later value.

## Parts

| Part | Decision | Unit | Why |
| --- | --- | --- | --- |
| Per-ticket event loop, ordering, exactly-once by event ID | host | `index.ts:94-149` | Concurrency and durability. |
| Ticket store: one JSON file per ticket, atomic replace | host | `TicketStore`, `index.ts:49-66` | Exact durability. |
| Resume after restart: arm the deadline of every waiting ticket | host | `start`, `index.ts:162-167` | Mechanism. |
| Request shape and ID checks | crisp | `server.ts:44-46`, `index.ts:171` | Exact, no meaning involved. |
| HTTP routes, JSON bodies | service | `server.ts:27-58` | The outside world. |
| `message`, `reply`, `close` transitions: status, draft dropped, due cleared | host | `index.ts:102-130` | The state machine; every status change is an exact rule over events. |
| When the customer started waiting: first customer message since an agent last answered | crisp | `waitingSince`, `index.ts:70-74` | Pure arithmetic over messages. |
| Urgency: low, normal, urgent | fn, decision | `triage/urgency` | A finite judgment with three spelled definitions. Today it shares one call with topic and summary. |
| Topic (two to four words) and one-sentence summary | fn | `triage/summarize` | Prose generation, a different task from the urgency decision. |
| Triage runs on every new message with the whole conversation | host | `index.ts:107-109` | Follow-up work after a commit (`context.after`), applied in conversation order by `through`. |
| Triage replaces an older triage only when it read more messages | host | `index.ts:113` | An exact ordering rule. |
| Missing details: what the agent still needs to ask | fn | `reply/missingDetails` | Today a hidden sub-task of `draft_reply.nl:9-10` ("ask for exactly the details still needed"). Small models do it better as a list they then write from. |
| Draft reply | fn | `reply/draft` | Prose from the conversation, the summary and the missing details. |
| Draft applies only to the conversation it read | host | `index.ts:125` | Same `through` rule as triage. |
| Response deadline: how long this ticket may wait | pluggable | `deadline/deadlineFor` (crisp default `RESPONSE_TIMES`) | A policy the desk owner states in words ("urgent: 15 minutes; billing: 1 hour"). Crisp table by urgency is the default; natural language reads a written service policy. |
| Deadline arithmetic: `since + minutes` | crisp | `index.ts:153-156` | Exact. |
| Is the ticket overdue now | host | `wake` case, `index.ts:132` | Timer comparison against `context.now`. |
| What an escalation does: whom to notify, the note, an optional holding reply | fn | `escalate/plan` | Today a callback (`onEscalate`, `index.ts:134`, `server.ts:66`) that prints one line. The plan is data; paging is the service. |
| Deliver the escalation plan | service | `onEscalate` callback | The outside world. |
| Agents' queue order | pluggable | `inbox/rank` (crisp default is the current sort, `index.ts:189-190`) | A ranking policy ("escalated first, then by deadline") written as a comparator. Natural language can weigh summaries; crisp checks the result is a permutation of the open tickets. |
| Customer text enters model calls | type | `Untrusted<string>` on `Message.text` | The runtime fences it; replaces any guard sentence. |

## Natural-language functions, step by step

Each function below is a `.nl` file with the typed contract shown. The steps are the algorithm the instructions
spell out.

### `triage/urgency`

```
args: messages: { from: "customer" | "agent", text: Untrusted<string> }[]
returns: "low" | "normal" | "urgent"
```

1. Find the customer messages that come after the last agent message. These are the open request.
2. List what the customer says has happened: what stopped working, what was lost, what was reported.
3. Answer "urgent" when the list contains one of: the product cannot be used at all; money or data is being lost; a
   security problem is reported.
4. Answer "low" when the open request is a question, a suggestion or praise and waiting a day changes nothing for
   the customer.
5. Answer "normal" for every other open request.

The decision judges events the customer reports. How strongly the customer writes is not an event.

### `triage/summarize`

```
args: messages, urgency
returns: { topic: Is<string, "two to four words naming a product area">,
           summary: Is<string, "one sentence that states what is wrong or wanted and what the customer already tried"> }
```

1. Read the open request (customer messages after the last agent message) and the earlier messages for context.
2. Name the product area in two to four words.
3. Write one sentence: what is wrong or wanted.
4. Append what the customer already tried, when they say so, in the same sentence.

### `reply/missingDetails`

```
args: messages, summary
returns: string[]   // each entry is one detail the agent needs
```

1. Read `summary` and the open request.
2. List what an agent needs to resolve the request and the conversation does not yet state (version, account,
   steps taken, error text).
3. Drop every entry the conversation already answers.
4. Return the remaining entries, most important first. An empty list means the agent can answer now.

### `reply/draft`

```
args: messages, summary, missing: string[]
returns: Is<string, "two to five sentences of plain text addressed to the customer">
```

1. If the conversation lets you answer, write the answer first, in one or two sentences.
2. Ask for each entry of `missing`, one question each, in the same order.
3. State only commitments the conversation supports (a refund the agent offered, a fix already confirmed).
4. Address the customer directly; write two to five sentences of plain text.

### `deadline/deadlineFor` (natural-language side of the pluggable)

```
args: urgency, topic, policy: string   // the desk's written service policy
returns: Is<number, "a positive whole number of minutes">
```

1. Read `policy`; find the rule that names `urgency`, then any rule that names `topic`.
2. When both name a time, use the shorter.
3. When none names a time, return the policy's stated default.

The crisp side is `RESPONSE_TIMES[urgency]` (`index.ts:44`). The setting `deadlineMode` selects `crisp`, `nl` or
`shadow`; shadow serves crisp and records agreement.

### `escalate/plan`

```
args: ticket summary (urgency, topic, summary), waited_minutes, messages
returns: { notify: "supervisor" | "on-call", note: string,
           holding_reply: string | null }   // text for the agent to send, never sent by the app
```

1. State in one line how long the customer has waited and the deadline that passed.
2. Choose "on-call" when urgency is urgent, otherwise "supervisor".
3. Write `note`: the summary, the topic and what is still missing, in two sentences.
4. When the customer has written more than once since the deadline passed, write `holding_reply` (two sentences that
   acknowledge the wait and give the next step). Otherwise null.

### `inbox/rank` (natural-language side of the pluggable)

```
args: tickets: { id, urgency, topic, summary, escalated, due }[]
returns: Is<string[], "every offered id exactly once">
```

1. Place escalated tickets first.
2. Order the rest so that the nearest `due` comes first; break ties by `urgency`, then by id.
3. Return the ids.

Crisp checks the permutation. The crisp default is the current comparator; the natural-language side exists so a
desk can state ranking rules in words (for example "security reports before everything").

## Pluggable points

| Point | Setting | Crisp default | Natural language | Verifier |
| --- | --- | --- | --- | --- |
| Deadline | `deadlineMode` | `RESPONSE_TIMES` by urgency | `deadlineFor` over a written policy | positive integer minutes; the commit uses `since + minutes` |
| Inbox order | `inboxMode` | the comparator at `index.ts:189-190` | `rank` | permutation of open ticket ids |

Neither is hot (deadline: once per triage; inbox: once per GET). Pluggable is for the shadow comparison and so a desk
can change policy without code. Both settings use the shared vocabulary `crisp | nl | shadow`.

## Refinement candidates

Status: all `open`. "Crisp" means the runtime can check the predicate exactly; "judged" means a model judge checks
it (plans/REFINEMENT_TYPES.md).

| Slot | Proposed type | Check |
| --- | --- | --- |
| `Message.text` | `Untrusted<string>` | crisp (marking) |
| `triage/urgency` result | the `Urgency` union (already an enum) | crisp |
| `Triage.topic` | `Is<string, "two to four words naming a product area">` | crisp for the word count; judged for "product area" |
| `Triage.summary` | `Is<string, "one sentence that states what is wrong or wanted and what the customer already tried">` | judged |
| `Ticket.due` | `Is<number \| null, "null unless the ticket is open and a customer message has no later agent reply">` | crisp (`waitingSince`) |
| `Ticket.triaged` | `Is<number, "at most the number of messages">` | crisp |
| `Followup.through` | `Is<number, "at most the number of messages at send time">` | crisp |
| `draft` | `Is<string, "two to five sentences of plain text addressed to the customer">` | crisp for sentence count (approximate) and no markup; judged for "addressed to the customer" |
| `missingDetails` entries | `Is<string[], "each entry names a detail the conversation does not state">` | judged |
| `deadlineFor` result | `Is<number, "a positive whole number of minutes">` | crisp |
| `rank` result | `Is<string[], "every offered id exactly once">` | crisp |
| `escalate/plan.holding_reply` | `Is<string \| null, "null unless the customer wrote after the deadline">` | crisp |

## Model-facing changes needing live measurement

Each is sampled on the live executor (about 48 samples per variant) and logged in `plans/MODEL_FACING_CHANGES.md`.

1. **Split `triage.nl` into `urgency` and `summarize`.** Compare agreement of urgency with the bundled call on
   recorded conversations, and summary quality with the same judge. Hypothesis: urgency accuracy rises because the
   decision is asked on its own.
2. **Rewrite "Judge by what the customer says has happened, not by how strongly they say it"** (`triage.nl:15`) as
   "Judge by the events the customer reports." Same hypothesis check.
3. **Split `draft_reply.nl` into `missingDetails` and `draft`.** Compare drafts that ask for unneeded details, and
   drafts that omit needed ones.
4. **Replace "promise nothing the conversation does not support"** (`draft_reply.nl:9-10`) by the positive step 3
   above.
5. **`Untrusted<string>` rendering** of customer text (already logged as one unmeasured change for logs and wiki;
   add helpdesk to that entry).
6. **`escalate/plan`** is new text; measure the note and holding reply on recorded overdue tickets.

## As built

- Files: `types.ts` (shared types and the `Checked*` aliases), `refinements.ts` (crisp checkers; the compiled `.nl`
  modules register them), `triage/urgency.nl`, `triage/summarize.nl`, `reply/missingDetails.nl`, `reply/draft.nl`,
  `deadline/deadlineFor.nl`, `escalate/plan.nl`, `inbox/rank.nl`. `triage.nl` and `draft_reply.nl` are gone.
- Settings (`DeskOptions`, and `--deadline-mode`, `--inbox-mode`, `--escalation-mode`, `--service-policy FILE`,
  `--ranking-policy FILE` on the server target): `deadlineMode` and `inboxMode` default to `crisp` (today's behaviour);
  `shadow` serves the crisp side and records `pluggable_shadow` events. `escalationMode` defaults to `nl`.
- Deadline units: the pluggable returns milliseconds; the `nl` side multiplies `deadlineFor`'s minutes by 60000. The
  allowed wait is decided in the triage follow-up and travels in the `triaged` event (`Ticket.allowedMs`), so `reduce`
  stays synchronous arithmetic: `due = waitingSince + (allowedMs ?? normal)`.
- Crisp floors: a deadline or ranking that the `nl` side gets wrong (not a positive number, not a permutation of the
  open ticket ids) is replaced by the crisp answer and reported through `onFailure`; a failed escalation plan is replaced
  by `crispEscalationPlan` (pager by urgency, the summary as note, no holding reply), so a page is never lost to a model
  failure.
- `rank` and `escalate/plan.holding_reply` need facts outside their value, so the desk checks them in code
  (`isRanking`; a holding reply is kept only when `late_messages` is 2 or more) instead of with `Is<...>`.
- Ticket gained `missing`, `allowedMs` and `escalation`; stored tickets without them are filled on load. `onEscalate`
  receives `(ticket, plan)`. The plan reaches the ticket through a `planned` follow-up event, after the pager callback.
- Refinements adopted with crisp checkers: topic (two to four words; "product area" is not checked), summary (one
  sentence), draft (two to five sentences of plain markdown-free text), deadline minutes. The judged candidates
  (`missingDetails` entries) and the context-bound ones stay open; a judged predicate would add a model call per value.

## Questions for the owner

Decided on 2026-10-09 when the owner delegated the review (answers kept as built):
1. The holding reply stays data for an agent to send; the desk never sends it.
2. The deadline is pluggable, with the table as the default.
3. Authentication stays outside this app (`server.ts` trusts its callers).

First-draft questions, for the record:

1. Should an escalation send the holding reply automatically? This document keeps it as data for the agent to send,
   because the desk's customers should not receive model text without an agent.
2. Is the deadline pluggable worth it, or is a settings table enough? The table is the default either way.
3. `server.ts` trusts its callers (`server.ts:3`). Authentication stays outside this app; confirm.
