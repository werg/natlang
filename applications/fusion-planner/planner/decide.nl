---
args:
  edge: EdgeFacts
  problem?: string
returns: EdgeDecision
---
Decide whether the value of one edge is handed from the producer to the consumer as a Neuralese block ("fuse") or as text ("keep-text"). Fusing is allowed only when every condition holds. Take the steps in order and stop at the first one that says keep-text.

1. Readers. edge.readers lists everyone who reads the value. Only one reader may exist, and it must have kind "consumer" and certain true. If there is any other reader, or the consumer reader has certain false, answer keep-text. The reason names the kind and detail of the first reader that is not a certain consumer.
2. Model. edge.producer.model and edge.consumer.model must be equal; two null values are equal. If they differ, answer keep-text and name both.
3. Types. When edge.alreadySoft is true, answer keep-text. When edge.consumer.param is null, answer keep-text. When edge.consumer.paramType is not null and is not the same text as edge.type, answer keep-text and name both types.
4. Judgment. need = needsReading(edge). When need.needed is true, answer keep-text with need.reason. When edge.finite is true, answer keep-text: the value has only a few possible values, so its text costs about one token.
5. Otherwise answer fuse. The reason says that one consumer reads the value and both sides share the model.

problem, when given, is why an earlier answer for this edge was rejected. Do not answer fuse for an edge that problem says cannot be fused. Return { edge: edge.id, decision, reason } with a reason of one sentence.
