---
description: System entry planning (pi-durable prompt.ts planSystemEntries, planSections, planTools, systemEntry; pi-ai declarationsEqual, toToolDeclaration; spec §7.4). The pi.system entries that make the transcript's replayed sections and tools equal the desired ones, in values and order.
args:
  view: ContextView
  desired: "{ key: string, text: string }[]"
  tools: AgentTool[]
  now: number
returns: "{ message: SystemMessage, edits?: ContextEdit[] }[]"
---
Plan the system entries one request needs: what the model is told about prompt and tool changes. desired is the
rendered sections in order; tools are the agent's tools in order; view.sections and view.tools are what the
transcript shows now. Compute it exactly in eval. Every entry's message is { role: "system", content: "", sections?,
toolsRemoved?, toolsAdded?, timestamp: now }: include sections only when given, toolsRemoved and toolsAdded only when
non-empty. A declaration written into the transcript has only name, description, parameters, and constrainedSampling
when the tool has it; two declarations are equal when the JSON of those four fields is equal.

1. Rebaseline after a head marker. When view.head is not null and no entry of view.entries with kind "pi.system" has
   an id greater than view.head.id: return exactly one entry, a complete baseline, even if it restates what is shown:
   sections = an object with every desired key -> text in order (possibly empty {}); toolsAdded = every tool's
   declaration in order. Its edits are { target: id, action: "omit" } for every "pi.system" entry in view.entries
   (leave edits out when there are none).
2. Otherwise section patches. shown = view.sections. patchedOrder = the shown keys that are desired (in shown order),
   then the desired keys not shown (in desired order).
   - If patchedOrder differs from the desired key order at any position: two patches: first every shown key -> null,
     then every desired key -> text, in desired order.
   - Otherwise one minimal patch: each shown key whose desired text differs -> the new text, or null when it is no
     longer desired; then each desired key not shown -> its text. An empty patch means no patch.
   patches = that list (0, 1 or 2 objects).
3. Tool changes. offered = view.tools. kept = the offered declarations whose name is among tools and equal to that
   tool's declaration, in offered order. added = the tools whose name is not in kept, in tools order.
   - If kept followed by added does not have exactly the tools' name order: toolsRemoved = every offered name (as
     { name }), toolsAdded = every tool's declaration, in order.
   - Otherwise toolsRemoved = the offered names not kept, toolsAdded = added's declarations.
4. Entries.
   - No tool changes (both lists empty): one entry per patch, each with that patch as sections.
   - Tool changes and no patch: one entry with only the tool changes.
   - Both: one entry per patch; the last one also carries the tool changes.
Return the entries as { message } objects (and edits only in case 1).

Return a result that is not null from eval, exactly as computed: end with an eval whose code is `return <the variable that holds it>;` and set finish true. Never write it out in return_result: it carries model text and provider data that must stay byte for byte.
