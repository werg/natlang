# Read-before-code probe: sample problems

These eight cases illustrate the 48-case [IR corpus](../data/teacher/read-before-code-probe.ir.jsonl). The agent sees the function instructions and typed scope. Lazy input values must be inspected with `read_value` or `eval`; folder contents must be read through file tools or `fs` in `eval`. Expected returns are oracle data, not shown to the agent.

## read-before-code:input_numeric-00

Inspect brief/memo and brief/values before choosing the computation.
Write TypeScript in eval that implements the memo for these values.
Return the resulting number array.

```json
{
  "brief": {
    "memo": "Only the even measurements are eligible. Square each eligible value, retaining source order.",
    "values": [
      -3,
      0,
      2,
      5,
      2,
      8,
      -1
    ]
  }
}
```

**Expected return**

```json
[0,4,4,64]
```

## read-before-code:input_numeric-05

Inspect brief/memo and brief/values before choosing the computation.
Write TypeScript in eval that implements the memo for these values.
Return the resulting number array.

```json
{
  "brief": {
    "memo": "Use each distinct positive measurement once, in the order of its first appearance, then double it.",
    "values": [
      0,
      6,
      6,
      -1,
      2,
      4,
      2
    ]
  }
}
```

**Expected return**

```json
[12,4,8]
```

## read-before-code:input_text-00

Inspect brief/memo and brief/entries before deciding how to process the entries.
Write the selected processing code in eval. Each entry is id|action|owner.
Return the selected ticket ids.

```json
{
  "brief": {
    "memo": "Keep only entries that start with an action verb (ship, call, or review), ignoring case. Return their ticket ids in source order.",
    "entries": [
      "T1|Ship replacement|Ada",
      "T2|wait for reply|Bo",
      "T3|CALL vendor|Cy",
      "T4|review traces|Ada"
    ]
  }
}
```

**Expected return**

```json
["T1","T3","T4"]
```

## read-before-code:input_text-05

Inspect brief/memo and brief/entries before deciding how to process the entries.
Write the selected processing code in eval. Each entry is id|action|owner.
Return the selected ticket ids.

```json
{
  "brief": {
    "memo": "Keep the first ticket for each distinct owner, ignoring owner case. Return those ticket ids in first-seen order.",
    "entries": [
      "T5|wait|Bo",
      "T6|review|Cy",
      "T7|call|ada",
      "T8|ship|Dee"
    ]
  }
}
```

**Expected return**

```json
["T5","T6","T7","T8"]
```

## read-before-code:file_table-00

Read policy.md and accounts.json in the folder before deciding what code to write.
Use eval to implement the policy against the file contents.
Return the selected account ids. Leave the files unchanged.

**policy.md**

```text
Include active accounts with balance at least 50. Return their ids in file order.
```

**accounts.json**

```text
[
  {
    "id": "A1",
    "owner": "Ada",
    "region": "west",
    "status": "active",
    "balance": 70,
    "opened": 2025
  },
  {
    "id": "A2",
    "owner": "Bo",
    "region": "east",
    "status": "closed",
    "balance": -8,
    "opened": 2024
  },
  {
    "id": "A3",
    "owner": "Ada",
    "region": "north",
    "status": "active",
    "balance": 50,
    "opened": 2026
  },
  {
    "id": "A4",
    "owner": "Cy",
    "region": "west",
    "status": "suspended",
    "balance": 90,
    "opened": 2023
  }
]
```

**Expected return**

```json
["A1","A3"]
```

## read-before-code:file_table-05

Read policy.md and accounts.json in the folder before deciding what code to write.
Use eval to implement the policy against the file contents.
Return the selected account ids. Leave the files unchanged.

**policy.md**

```text
For each owner, keep only the account with the highest balance; break ties by earlier file order. Return selected ids in owner first-seen order.
```

**accounts.json**

```text
[
  {
    "id": "B1",
    "owner": "Cy",
    "region": "south",
    "status": "active",
    "balance": 40,
    "opened": 2024
  },
  {
    "id": "B2",
    "owner": "Ada",
    "region": "west",
    "status": "active",
    "balance": 120,
    "opened": 2026
  },
  {
    "id": "B3",
    "owner": "Bo",
    "region": "east",
    "status": "active",
    "balance": -5,
    "opened": 2025
  },
  {
    "id": "B4",
    "owner": "Cy",
    "region": "south",
    "status": "closed",
    "balance": 55,
    "opened": 2025
  }
]
```

**Expected return**

```json
["B4","B2","B3"]
```

## read-before-code:file_log-00

Read criteria.md and events.json in the folder before choosing the analysis code.
Use eval to implement the criteria against the events.
Return the selected event ids. Leave the files unchanged.

**criteria.md**

```text
Report ids of failed requests from the api service, excluding failures explicitly marked as expected.
```

**events.json**

```text
[
  {
    "id": "L1",
    "minute": 3,
    "service": "api",
    "level": "fail",
    "note": "unexpected"
  },
  {
    "id": "L2",
    "minute": 8,
    "service": "security",
    "level": "warn",
    "note": "scan"
  },
  {
    "id": "L3",
    "minute": 10,
    "service": "api",
    "level": "ok",
    "note": "recovered"
  },
  {
    "id": "L4",
    "minute": 12,
    "service": "security",
    "level": "warn",
    "note": "token"
  },
  {
    "id": "L5",
    "minute": 13,
    "service": "worker",
    "level": "fail",
    "note": "expected"
  },
  {
    "id": "L6",
    "minute": 15,
    "service": "api",
    "level": "fail",
    "note": "unexpected"
  }
]
```

**Expected return**

```json
["L1","L6"]
```

## read-before-code:file_log-05

Read criteria.md and events.json in the folder before choosing the analysis code.
Use eval to implement the criteria against the events.
Return the selected event ids. Leave the files unchanged.

**criteria.md**

```text
For each service with any failure, report the id of its first failure. Preserve the order in which services first fail.
```

**events.json**

```text
[
  {
    "id": "M1",
    "minute": 2,
    "service": "worker",
    "level": "warn",
    "note": "queue"
  },
  {
    "id": "M2",
    "minute": 10,
    "service": "security",
    "level": "warn",
    "note": "login"
  },
  {
    "id": "M3",
    "minute": 11,
    "service": "worker",
    "level": "fail",
    "note": "unexpected"
  },
  {
    "id": "M4",
    "minute": 14,
    "service": "api",
    "level": "fail",
    "note": "expected"
  },
  {
    "id": "M5",
    "minute": 18,
    "service": "api",
    "level": "fail",
    "note": "unexpected"
  },
  {
    "id": "M6",
    "minute": 19,
    "service": "worker",
    "level": "ok",
    "note": "recovered"
  }
]
```

**Expected return**

```json
["M3","M4"]
```
