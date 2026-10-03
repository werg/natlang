---
name: sql-join-keys
description: Chooses join keys for text-to-SQL over unfamiliar schemas, preferring declared foreign keys and avoiding joins on display names. Use before writing a query that joins two or more tables.
metadata:
  natlang: |
    scope:
      maxJoins:
        type: number
        value: 3
    requires:
      services: [sql]
---

1. List the tables the question needs.
2. Join on declared foreign keys first (`references/schema-notes.md`), then on columns named `<table>_id`.
3. Never join on names or titles; they are not unique.
4. Keep the number of joins at most `maxJoins` unless the question needs more.
