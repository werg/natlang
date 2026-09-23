# Read-before-code probe: sample problems

These ten cases illustrate the 59-case [IR corpus](../data/teacher/read-before-code-probe.ir.jsonl). The agent sees the function instructions and typed scope. Large ordinary input batches show only their first few rows in the opening preview. Folder contents must be read through file tools or `fs` in `eval`. Expected returns are oracle data, not shown to the agent.

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

## read-before-code:input_accounts-00

Examine the complete accounts batch before deciding how to implement the request.
Write the selected account processing code in eval.
Return the requested account ids.

```json
{
  "request": "Include active accounts with balance at least 50. Return their ids in file order.",
  "accounts": [
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
    },
    {
      "id": "A5",
      "owner": "Bo",
      "region": "west",
      "status": "active",
      "balance": 35,
      "opened": 2025
    },
    {
      "id": "A6",
      "owner": "Cy",
      "region": "north",
      "status": "active",
      "balance": -6,
      "opened": 2026
    },
    {
      "id": "A7",
      "owner": "Dee",
      "region": "south",
      "status": "active",
      "balance": 50,
      "opened": 2024
    },
    {
      "id": "A8",
      "owner": "Ada",
      "region": "east",
      "status": "suspended",
      "balance": 130,
      "opened": 2025
    },
    {
      "id": "A9",
      "owner": "Bo",
      "region": "east",
      "status": "active",
      "balance": 70,
      "opened": 2026
    },
    {
      "id": "A10",
      "owner": "Dee",
      "region": "west",
      "status": "closed",
      "balance": -9,
      "opened": 2023
    },
    {
      "id": "A11",
      "owner": "Eli",
      "region": "north",
      "status": "active",
      "balance": 70,
      "opened": 2025
    },
    {
      "id": "A12",
      "owner": "Cy",
      "region": "south",
      "status": "active",
      "balance": 55,
      "opened": 2026
    }
  ]
}
```

**Expected return**

```json
["A1","A3","A7","A9","A11","A12"]
```

## read-before-code:input_accounts-05

Examine the complete accounts batch before deciding how to implement the request.
Write the selected account processing code in eval.
Return the requested account ids.

```json
{
  "request": "For each owner, keep only the account with the highest balance; break ties by earlier file order. Return selected ids in owner first-seen order.",
  "accounts": [
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
    },
    {
      "id": "B5",
      "owner": "Eli",
      "region": "west",
      "status": "active",
      "balance": 70,
      "opened": 2025
    },
    {
      "id": "B6",
      "owner": "Bo",
      "region": "east",
      "status": "suspended",
      "balance": -11,
      "opened": 2026
    },
    {
      "id": "B7",
      "owner": "Ada",
      "region": "north",
      "status": "active",
      "balance": 120,
      "opened": 2024
    },
    {
      "id": "B8",
      "owner": "Dee",
      "region": "south",
      "status": "active",
      "balance": 50,
      "opened": 2025
    },
    {
      "id": "B9",
      "owner": "Cy",
      "region": "west",
      "status": "closed",
      "balance": -3,
      "opened": 2026
    },
    {
      "id": "B10",
      "owner": "Dee",
      "region": "north",
      "status": "active",
      "balance": 80,
      "opened": 2026
    },
    {
      "id": "B11",
      "owner": "Bo",
      "region": "west",
      "status": "active",
      "balance": 90,
      "opened": 2025
    },
    {
      "id": "B12",
      "owner": "Eli",
      "region": "east",
      "status": "active",
      "balance": 25,
      "opened": 2024
    }
  ]
}
```

**Expected return**

```json
["B4","B2","B11","B5","B10"]
```

## read-before-code:input_events-00

Examine the complete event batch before deciding how to implement the request.
Write the selected event analysis in eval.
Return the requested event ids.

```json
{
  "request": "Report ids of failed requests from the api service, excluding failures explicitly marked as expected.",
  "events": [
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
    },
    {
      "id": "L7",
      "minute": 17,
      "service": "worker",
      "level": "warn",
      "note": "queue"
    },
    {
      "id": "L8",
      "minute": 18,
      "service": "security",
      "level": "fail",
      "note": "unexpected"
    },
    {
      "id": "L9",
      "minute": 19,
      "service": "api",
      "level": "warn",
      "note": "latency"
    },
    {
      "id": "L10",
      "minute": 20,
      "service": "worker",
      "level": "ok",
      "note": "recovered"
    },
    {
      "id": "L11",
      "minute": 21,
      "service": "security",
      "level": "ok",
      "note": "recovered"
    },
    {
      "id": "L12",
      "minute": 22,
      "service": "api",
      "level": "fail",
      "note": "expected"
    }
  ]
}
```

**Expected return**

```json
["L1","L6"]
```

## read-before-code:input_events-05

Examine the complete event batch before deciding how to implement the request.
Write the selected event analysis in eval.
Return the requested event ids.

```json
{
  "request": "For each service with any failure, report the id of its first failure. Preserve the order in which services first fail.",
  "events": [
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
    },
    {
      "id": "M7",
      "minute": 21,
      "service": "security",
      "level": "fail",
      "note": "unexpected"
    },
    {
      "id": "M8",
      "minute": 22,
      "service": "api",
      "level": "warn",
      "note": "latency"
    },
    {
      "id": "M9",
      "minute": 23,
      "service": "security",
      "level": "warn",
      "note": "token"
    },
    {
      "id": "M10",
      "minute": 24,
      "service": "worker",
      "level": "fail",
      "note": "expected"
    },
    {
      "id": "M11",
      "minute": 25,
      "service": "api",
      "level": "ok",
      "note": "recovered"
    },
    {
      "id": "M12",
      "minute": 26,
      "service": "security",
      "level": "fail",
      "note": "unexpected"
    }
  ]
}
```

**Expected return**

```json
["M3","M4","M7"]
```

## read-before-code:input_incidents-00

Read the incident notes, then interpret the request using their meaning and evidence.
Write the resulting selection in eval.
Return the matching note ids in source order.

```json
{
  "request": "Which notes confirm that the reported problem has recovered? Require actual user or operational evidence, rather than a plan or an unverified claim.",
  "notes": [
    {
      "id": "A1",
      "note": "Finance confirmed a second settled payment for the same order; the customer was charged twice."
    },
    {
      "id": "A2",
      "note": "For a separate order, the second bank entry is only a pending authorization and has not settled."
    },
    {
      "id": "A3",
      "note": "The reporter cleared the cache and can now open the dashboard again."
    },
    {
      "id": "A4",
      "note": "The operator intends to roll back once traffic drains, but has not started."
    },
    {
      "id": "A5",
      "note": "Rollback completed and the error rate has stayed at baseline for twenty minutes."
    },
    {
      "id": "A6",
      "note": "The blank page reproduces in staging; production users are unaffected."
    },
    {
      "id": "A7",
      "note": "Production checkout is rejecting cards, and monitoring confirms seventy failed attempts."
    },
    {
      "id": "A8",
      "note": "Support thinks the incident may have recovered; no telemetry or user retest is available."
    },
    {
      "id": "A9",
      "note": "The vendor reports that its webhook backlog cleared, but our queue is still growing."
    },
    {
      "id": "A10",
      "note": "The patch is deployed and smoke checks pass, yet the customer still sees the failure."
    },
    {
      "id": "A11",
      "note": "Key rotation restored service; monitors and the affected user both confirm recovery."
    },
    {
      "id": "A12",
      "note": "The suspicious integration was disabled, while the other tenants continue normally."
    }
  ]
}
```

**Expected return**

```json
["A3","A5","A11"]
```

## read-before-code:input_incidents-05

Read the incident notes, then interpret the request using their meaning and evidence.
Write the resulting selection in eval.
Return the matching note ids in source order.

```json
{
  "request": "Which notes describe a remedy that is only planned or queued and has not been applied yet?",
  "notes": [
    {
      "id": "B1",
      "note": "For one renewal, a duplicate invoice was drafted but voided before capture; the ledger shows one payment."
    },
    {
      "id": "B2",
      "note": "For a different renewal, two captures posted to the ledger, confirmed by finance."
    },
    {
      "id": "B3",
      "note": "A fix is queued for tomorrow; no deployment has happened."
    },
    {
      "id": "B4",
      "note": "The on-call engineer reverted the release and both health checks and customer retries pass."
    },
    {
      "id": "B5",
      "note": "A test tenant saw failed logins in staging; live tenant traffic is healthy."
    },
    {
      "id": "B6",
      "note": "Live tenants cannot upload files, confirmed in access logs and support reports."
    },
    {
      "id": "B7",
      "note": "The provider marked the incident resolved, but our failed uploads are still increasing."
    },
    {
      "id": "B8",
      "note": "A restart finished, though there has been no customer retest and metrics remain unavailable."
    },
    {
      "id": "B9",
      "note": "The queue drained after the restart, and both telemetry and a user confirm uploads work."
    },
    {
      "id": "B10",
      "note": "The patch passed tests but has not reached production."
    },
    {
      "id": "B11",
      "note": "Synthetic checks look green, but a customer reproduced the error after deployment."
    },
    {
      "id": "B12",
      "note": "The affected integration was switched off, preventing further bad events."
    }
  ]
}
```

**Expected return**

```json
["B3","B10"]
```
