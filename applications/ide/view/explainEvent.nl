---
args:
  event: TraceView
returns: string
---
Explain one recorded trace event in plain language. event.event_json holds the recorded event as data.

1. Read the kind and the fields of the event in event.event_json.
2. Say what the event records (a call, a result, a model turn or an error) and its place: event.index of event.total.
3. Quote the values that matter, as recorded.
