---
description: Advisory review of one source item at intake. Recommends admit or hold; the registry changes only by an explicit decision.
args:
  item: SourceItem
  precedents: Precedent[]
returns: ItemRecommendation
types:
  SourceItem: { dataset: string, id: string, visible: Untrusted<string>, annotated_label: Untrusted<string>, contract: string, answer_format: string | null }
  Precedent: { id: string, concern: string, reason: string }
  Concern: "none" | "label-disagrees-with-source" | "unstated-premise" | "ambiguous-question" | "contract-mismatch" | "needs-context"
  Quote: { quote: string }
  ProposedEntry: { reason: string }
  ItemRecommendation: { recommendation: "admit" | "hold", concern: Concern, reason: Is<string, "one to three sentences naming what the visible text states, lacks or contradicts">, evidence: Quote[], proposed_entry: ProposedEntry | null, confidence: "high" | "medium" | "low" }
---
Recommend whether the source item item belongs in the pool of its family. A person decides afterwards; your recommendation is advice.

1. Read item.contract (the question or criterion the family asks about this item) and item.visible.
2. Write in one phrase the answer that item.visible supports under item.contract.
3. Compare that answer with item.annotated_label. When they agree, go to step 4. When they differ, set concern "label-disagrees-with-source", copy the sentence of item.visible that supports your answer into evidence, and go to step 6.
4. List the facts that item.annotated_label needs. Mark each one as stated in item.visible or not stated. When a needed fact is not stated, set concern "unstated-premise" and name that fact in reason.
5. Read the question once more. When two readings give different labels, set concern "ambiguous-question" and write both readings in reason. When the label depends on information that item.visible does not carry (a sender's context, a document outside the item), set concern "needs-context". When item.answer_format or item.contract asks for something the label does not provide, set concern "contract-mismatch".
6. When an entry of precedents has the same concern, write reason in the wording style of that entry.
7. Return recommendation "hold" when a concern was set, with proposed_entry { reason } carrying the same reason. Otherwise return recommendation "admit", concern "none" and proposed_entry null.

Every quote in evidence is copied exactly from item.visible. Set confidence to "high" when the steps settled the concern without a judgement call, "medium" when one step needed a judgement call, and "low" otherwise.
