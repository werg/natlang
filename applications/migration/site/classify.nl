---
description: Migration site classification. How one site uses the old thing, and whether the migration must edit it or leave it.
args:
  intent: Intent
  site: Site
  evidence?: string
returns: CheckedUsage
---
Classify the use of intent.old in site.text. evidence, when given, is what a failing check or rejected patch said about
this site.

1. pattern. Take the first that fits:
   - "comment-or-doc": the mention is inside a comment or documentation text.
   - "string-literal": the mention is inside a string that is data (a message, a key, a path).
   - "import": it names the thing in an import list or require.
   - "export": it names the thing in an export or re-export.
   - "declaration": it defines the thing (function, class, variable, type, method, parameter).
   - "type-position": it is used as a type or in a type annotation.
   - "property-access": it is read as a member of another value (obj.old).
   - "call": it is called, constructed or applied.
   - "test": the site is in a test and asserts or exercises the old thing.
   - "unrelated": the text matches but means something else (a longer identifier that contains it, a different
     binding with the same name).
2. action. "edit" when the migration must change the site to meet intent.summary: declaration, import, export, call,
   type-position, property-access, and test sites that use the old name as code. "leave" for comment-or-doc (unless
   the request is about the text), string-literal (unless code looks the thing up by that string), and unrelated. With evidence that names this site as still using the old thing or as broken, action is "edit".
3. reason: one sentence naming the evidence in the site's text.

Return { pattern, action, reason }.
