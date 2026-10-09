---
args:
  facts: ComponentFacts
returns: string[]
---
Choose the component keys the next edit rewrites. facts.eligible lists the keys that training cases cover, each with coveredCases (cases that exercise it), failingCases (covered cases that scored below 1), proposals (edits already tried) and accepts (edits accepted).

1. Choose the key with the highest failingCases.
2. When several keys tie, choose the one with the smallest proposals, then the smallest key.
3. Return a list holding that key. The host adds the keys it depends on.
