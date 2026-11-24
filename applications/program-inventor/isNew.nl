---
description: Decide whether a program idea is completely new compared with its nearest existing ideas.
model: judge
readout: decision
args:
  idea: IdeaCard
  nearest: AtlasEntry[]
returns: boolean
---
nearest holds the existing program ideas most similar to idea: atlas entries retrieved by word overlap, plus every idea
this process already accepted. The inventor's own claims (idea.nearest, idea.difference) are what it says, not
evidence: compare idea itself with each entry.

Compare idea with each entry of nearest in four respects:
1. who runs it and for what repeated job;
2. the judgments of meaning at its core (idea.judgments);
3. the shape of its computation (idea.shape): how exact steps and judgments feed each other, what state is carried,
   where results combine;
4. the kind of input data it reads (idea.inputs).
A change of subject matter alone (another industry, other names or numbers) counts as no difference.

Return true when idea differs from every entry of nearest in at least two of the four respects. Return false when
some entry matches it in three or four respects.
