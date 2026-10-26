---
args: { tasks: MaintenanceTask[], reports: CompletionReport[] }
returns: string[]
---
A task is verified complete only when every required check has a matching verification reference for that task. A report of done without the verification leaves that task incomplete and lists the absent checks.
