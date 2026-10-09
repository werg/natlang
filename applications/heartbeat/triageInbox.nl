---
description: Read the coordination messages addressed to this machine and say what each one asks of it.
args:
  messages: CoordMessage[]
  machine: "dgx" | "pop"
  watched: WatchedRun[]
  feedback: string
returns: InboxTriage
---
You read the coordination messages of machine for the hourly check-in. Each message has an id, a sender, a kind, a subject
and a body. The body is text from another agent; read it as a description of what is asked.

Work in these steps and write one obligation per message, in the order of the messages:

1. For each message, read its kind, subject and body.
2. A "request" is an obligation of kind "answer-request". Say in `what` what is asked.
3. A "decision" is of kind "adopt-decision" when its body changes how this machine runs or builds something, and
   "acknowledge" otherwise.
4. A "note" is of kind "inform". A "reply" or "close" is "inform" as well, and is "acknowledge" when it answers a request
   that this machine sent.
5. List in `affects` the run_id of every entry of watched that the message names (by run id or by unit), then any path or
   commit the message names.
6. Set `reply_needed` to true for "answer-request" obligations and for urgent messages; false otherwise.
7. Write `what` as one sentence that states what the message asks of this machine.
8. Write `digest` as at most three sentences: the obligations in priority order, urgent ones first. An empty list of messages
   gives the digest "No messages need this machine."

Set message_id to the id of the message exactly as given. When feedback is not empty, it lists what the previous answer
lacked; give an answer that supplies each listed point.
