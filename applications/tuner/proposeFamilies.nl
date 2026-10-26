---
args: { discovery: string }
returns: { families: { name: string; members: string[] }[] }
---
You decide which groups of functions, from several programs, should share one learned behavior (a family). A family is trained once on the examples of all its members, and each member then reuses it. A good family groups functions that do the same kind of work, so learning from one helps the others. A bad family mixes different kinds of work, and then every member gets worse.

discovery is JSON:

- members: every function you may use, written "program/binding".
- similarities: pairs of functions from different programs, with cosine: how alike their learned adjustments are, from -1 (opposite) to 1 (the same). Above 0.5 is clearly alike; near 0 is unrelated.
- clusters: groups the automatic rule formed from pairs at or above its threshold.
- tags: groups the program authors declared, by family name.

Work in these steps:

1. Start from the author tags. Keep each tag as a family under its own name, unless its members have clearly negative similarities with each other; then leave that tag out.
2. Look at each cluster. Keep it when its members are alike to each other, not only through one shared neighbour: a chain where the first and last members have a low cosine should be split into the tighter groups.
3. Give each new family a short name of letters, digits, dashes or dots that says what its members do in common, if the names show it; otherwise "family-1", "family-2" and so on.
4. A function belongs to at most one family. A family needs at least two members.

Answer with families: the list of families, each with its name and its members written exactly as in members. When nothing is worth sharing, answer with an empty list.
