---
description: The edit tool's matching policy. Match every edit against the original content and apply them all, or name what is wrong, with pi's exact messages (pi edit-diff.ts applyEditsToNormalizedContent).
args:
  content: string
  edits: "{ oldText: string, newText: string }[]"
  path: string
returns: "{ base: string, content: string } | { error: string }"
---
content is the file's text with LF line endings. Apply edits to it as one change: every edit is matched against this
same original, never against the result of an earlier edit. text, in your folder, does the exact text work. Return
{ base, content }: base is the original content, content the new one. When a rule below fails, return
{ error: "<message>" } with that message exactly, and nothing else. n is edits.length; i is an edit's index in edits.
In the messages, "one:" applies when n is 1, "several:" otherwise.

1. LF: each edit's oldText and newText become text.normalizeToLF(...) of themselves. Use these from here on.
2. Empty: the first edit whose oldText is "" fails.
   one: "oldText must not be empty in <path>."  several: "edits[<i>].oldText must not be empty in <path>."
3. Which content to match in: find every edit once with text.fuzzyFindText(content, oldText). If any of those
   results has usedFuzzyMatch true, base = text.normalizeForFuzzyMatch(content) (the tolerant form); otherwise base
   is content itself.
4. For each edit, in order:
   a. m = text.fuzzyFindText(base, oldText). m.found false fails:
      one: "Could not find the exact text in <path>. The old text must match exactly including all whitespace and newlines."
      several: "Could not find edits[<i>] in <path>. The oldText must match exactly including all whitespace and newlines."
   b. k = text.countOccurrences(base, oldText). k > 1 fails:
      one: "Found <k> occurrences of the text in <path>. The text must be unique. Please provide more context to make it unique."
      several: "Found <k> occurrences of edits[<i>] in <path>. Each oldText must be unique. Please provide more context to make it unique."
   c. Keep the match { editIndex: i, matchIndex: m.index, matchLength: m.matchLength, newText }.
5. Overlap: sort the matches by matchIndex. Two neighbours overlap when the earlier's matchIndex + matchLength is
   greater than the later's matchIndex; the first such pair fails:
   "edits[<earlier editIndex>] and edits[<later editIndex>] overlap in <path>. Merge them into one edit or target disjoint regions."
6. Replace: with the tolerant base (step 3), new = text.applyReplacementsPreservingUnchangedLines(content, base,
   matches), which keeps untouched lines byte for byte; otherwise new = text.applyReplacements(base, matches). If
   either throws, return its message as the error.
7. No change: new equal to content fails.
   one: "No changes made to <path>. The replacement produced identical content. This might indicate an issue with special characters or the text not existing as expected."
   several: "No changes made to <path>. The replacements produced identical content."
8. Return { base: content, content: new }.
