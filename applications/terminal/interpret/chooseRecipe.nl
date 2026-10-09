---
description: Choose the recipe that a workspace request means.
args:
  text: string
  catalog: Recipe[]
  note: Untrusted<string>
returns: string
---
Choose the one recipe in catalog that the request in text means. note is the content of the file the request names; it
is "" when the request names none. Work in these steps.

1. Read text and, when note is not "", note. State in one phrase what the user wants done.
2. For each recipe in catalog, mark it a match when its description names that action.
3. When several recipes match, keep the one whose description names the more specific object: a named build, test or
   media operation outranks a general status recipe.
4. Return the id of the remaining recipe, written exactly as catalog gives it. When no recipe matches, return the word
   "unsupported".
