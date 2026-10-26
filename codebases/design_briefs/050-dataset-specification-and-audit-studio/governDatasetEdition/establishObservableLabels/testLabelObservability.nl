---
args: {"labelDescription": "string", "use": "DatasetUse", "events": "Evidence[]"}
returns: "ObservableLabel"
---
Identify exactly which information in the candidate record would let a qualified annotator determine this label. Distinguish an observable label from an inference needing absent history or private attributes. Test a counterexample with the same visible evidence but a different plausible hidden truth. If that counterexample cannot be distinguished, mark the label unobservable and require a task revision or explicit unknown disposition.
