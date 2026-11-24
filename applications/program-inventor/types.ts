/** One existing program idea in the atlas: a training task family, an application, a design brief, an example, a
 * dataset source, or an idea this process already accepted. */
export type AtlasEntry = {
  /** `<kind>:<name>`, e.g. `application:sem-merge` or `invented:tide-table-reconciler`. */
  id: string;
  kind: 'invented' | 'application' | 'example' | 'design-brief' | 'authored-family' | 'pool-family' | 'dataset';
  /** The idea's name as the atlas line starts with it (family, application or brief directory name). */
  name: string;
  /** One line: the idea's name, then what it does. */
  line: string;
  /** TF-IDF cosine similarity to the idea being judged, when the entry was retrieved as a near neighbour. */
  similarity?: number;
};

/** The atlas as the inventor reads it: one line per existing idea, grouped by where it lives. */
export type Atlas = {
  /** Markdown, one `- line` per entry. */
  text: string;
  entries: AtlasEntry[];
  /** The git commit the atlas was built at. */
  commit: string;
};

/** A semantic judgment the program makes: one narrow question of meaning with a typed answer. */
export type Judgment = {
  /** The function's name, e.g. `supersedes`. */
  name: string;
  /** The question it answers, in one sentence. */
  question: string;
  /** Its TypeScript result type, e.g. `boolean` or `"keep" | "drop" | "split"`. */
  returns: string;
  /** The wrong reading a careless or keyword-driven reader would give, and why the right one differs. */
  trap: string;
};

/** A new program idea, before it is built. */
export type IdeaCard = {
  /** Lowercase words joined by dashes; unique in the atlas. */
  slug: string;
  /** Two sentences: what the program does and for whom. */
  pitch: string;
  /** Who runs this program repeatedly, on what, and what they do today instead. */
  user: string;
  /** The judgments of meaning at the program's core; at least three, each a different question. */
  judgments: Judgment[];
  /** How data flows: which exact steps and judgments feed which, what state is carried, where results combine. */
  shape: string;
  /** The exact work TypeScript does (parsing, ordering, aggregation, state transitions, graph walks). */
  crispWork: string;
  /** Semantic-FP constructs the program will use, from: decide, std-collection, std-convergence, higher-order,
   * iterate-until, structural-recursion, ensures, refinement, untrusted, neuralese, folder-transaction, pluggable. */
  constructs: string[];
  /** What one input instance holds and how large it is (items, words). */
  inputs: string;
  /** The exact output type and how a checker decides it is right. */
  output: string;
  /** Why no rule over surface cues (keywords, positions, formats) reaches the right output. */
  whyKeywordsFail: string;
  /** The atlas entries closest to this idea, by the name each atlas line starts with. */
  nearest: string[];
  /** For each of those, how this idea differs in user, judgments, shape and input data. */
  difference: string;
};

/** An idea the round already turned down, with the reason, so the next attempt goes elsewhere. */
export type RejectedIdea = {
  slug: string;
  pitch: string;
  reason: string;
};

/** A requirement a built program must meet. `check: "crisp"` ones are decided by code (qualification reports);
 * `check: "judged"` ones by `meetsBar`. */
export type Requirement = {
  /** E.g. `S4`. */
  id: string;
  group: 'functional' | 'semantic' | 'complexity' | 'code-depth' | 'novelty';
  check: 'crisp' | 'judged';
  /** The requirement as a sentence a reviewer can hold the program to. */
  statement: string;
};

/** A built program as the reviewer reads it. */
export type ProgramView = {
  idea: IdeaCard;
  /** Program files by path: root `.nl`, `types.ts`, the callable folder's `.nl` and `.ts` files. */
  files: Record<string, string>;
  /** One input instance (the first), as JSON text. */
  instance: string;
  /** The crisp metrics of the program (function counts, call depth, constructs found, input size). */
  metrics: string;
};

/** What the qualification services found about a submission. */
export type Qualification = {
  /** Every crisp problem: check errors, unmet crisp requirements, replay or control failures. Empty when it passes. */
  problems: string[];
  /** The crisp metrics, as text for the reviewer. */
  metrics: string;
  /** Path of the packaged `natlang.program/2` cases. */
  cases: string;
};

/** The outcome of a live run of one instance by a teacher model. */
export type LiveRun = {
  /** The instance's inputs, as JSON text. */
  instance: string;
  /** The instance's expected answer, as JSON text. */
  expected: string;
  passed: boolean;
  /** The answer the teacher returned, as JSON text. */
  answer: string;
  /** The checker's issues with it. */
  issues: string[];
};

/** A disagreement between the expected answer and a teacher's live answer, for adjudication. */
export type Disagreement = {
  program: Record<string, string>;
  instance: string;
  expected: string;
  answer: string;
  issues: string[];
};

/** What one attempt produced. */
export type AttemptResult = {
  slug: string;
  status: 'accepted' | 'not-new' | 'idea-problems' | 'build-failed' | 'below-bar' | 'unsolvable';
  reason: string;
  /** The submission directory, when a program was built. */
  submission?: string;
};
