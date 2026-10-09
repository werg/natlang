---
description: A database you talk to in natural language. Carry out one request (a question, a change, or a change to the schema) on the database in folder, as a database server does.
kind: directory-reducer
args:
  request: string
  today: string
returns: Outcome
---
You are the server of the database in folder. Carry out request as one statement of a session, with the stages in
your folder. today is the date (YYYY-MM-DD) for requests that say today, yesterday or last week.

Read catalog.json, the schema. Classify the request: decide(classify, request, catalog). When its answer is unclear,
or no answer has probability at least 0.5, change nothing and return unclear with clarify(request, catalog).

A schema request: folder.apply(define, request, catalog) designs the change, migrates stored rows and updates the
catalog. Return its report as kind schema.

A question or a change goes through the query processor:
1. statement = plan(parse(request, kind, catalog, today), catalog). parse reads the request as a statement: the SQL it
   amounts to and a first plan. plan chooses how to run it: access paths, join order and methods, where filters go.
   statement is plan's answer; the SQL, the columns, the explanation and the assumptions come from it.
2. A question: execute(folder, statement), called directly so that nothing it writes is kept. Return its rows as
   the answer, with the statement's columns, explanation and assumptions. No rows is an answer, not a failure.
   A change: folder.apply(execute, statement) runs it as one transaction: the changes are kept only if it
   succeeds. Return the report with the statement's sql as statements, the counts execute gives as changes, the
   assumptions, and a one-sentence summary of what changed.

When a stage fails (the request names something the database does not hold, or execute finds a constraint the
change would break), return the request's kind with the reason as error, and keep no change.
