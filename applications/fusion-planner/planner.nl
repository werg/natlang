---
description: Plans which hand-offs between functions keep their value as a Neuralese block (fused) and which stay text. One decision per candidate edge.
args:
  edges: EdgeFacts[]
  problems?: Problem[]
returns: EdgeDecision[]
---
edges are the candidate hand-offs of a program: for each, a function (the producer) returns a value that another function (the consumer) receives. Decide each one, in the stages of your folder. Return one EdgeDecision per edge, in the order of edges, with the edge's id in `edge`.

1. For all edges at once: decide(edge, problem). problem is the entry of problems whose edge is the edge's id, when there is one, otherwise omitted. problems are answers a checker rejected earlier; decide must not give the same answer again.
2. Return the decisions in the order of edges. Do not add, drop or merge edges, and do not change a decision.
