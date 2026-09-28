/** Source annotation disputes stay out of generation and training until adjudicated. */
export type SourceReview = {
  dataset: string;
  id: string;
  aliases: readonly string[];
  text: string;
  annotatedLabel: string;
  status: 'pending' | 'resolved';
  reason: string;
};

/** IDs use model-visible text; aliases preserve the earlier text+label identity. */
export const SOURCE_REVIEWS: readonly SourceReview[] = [
  {
    dataset: 'banking77',
    id: 'a1658811502225e680d921d1ba4aa4492cc1b7225160f46c1cc360393d0f2fcd',
    aliases: ['a9a729504209367ae5cdc1fba30d835718922358d51a8f922b205513a6925935'],
    text: 'I did not get the cash that is showing up in my app.',
    annotatedLabel: 'cash_withdrawal_not_recognised',
    status: 'pending',
    reason: 'Cash not received does not establish whether the withdrawal was unrecognized or failed.',
  },
  {
    dataset: 'banking77',
    id: '36796a0f7fcdef9e1f89b825bb4150ac5d530bc865d3da2a6f5b7902298c2acb',
    aliases: ['20d4392399dd4a19d86356619825f146f3b7b3bbbfeaa626c493f50ae5abc002'],
    text: 'My app says I withdraw funds from my account through an ATM.',
    annotatedLabel: 'cash_withdrawal_not_recognised',
    status: 'pending',
    reason: 'The visible statement does not establish whether the recorded withdrawal is disputed.',
  },
  {
    dataset: 'banking77',
    id: 'e8201faf7e42751836cf4ae49618b262fa23d3c7cc3783a7ec877acda07c69fe',
    aliases: ['8036fcc7415fdfebff9c8c50b5afeb7886d3f39f67a328a46b4ef34caa9b1fd5'],
    text: 'How can I dispute a debit transaction?',
    annotatedLabel: 'direct_debit_payment_not_recognised',
    status: 'pending',
    reason: 'A disputed debit transaction does not establish direct debit or lack of authorization.',
  },
  ...[
    ['db49742d7d0f0a9986b4fe7c04a26f815228b6ce811fc45d0c4ef7eb9410da7a',
      "Ellie's car broke down near a tunnel.\nBy the time she left the area she counted 274 cars.\nEveryone had to evacuate the tunnel on foot.\nEllie decided to take a nap in her car.", 'a',
      'Neither hypothesis explains the stated counting; the missing connecting events leave the preference unsupported.'],
    ['921971912e8a89bbed73f18ab99389871fa1a3be590f5a5e1b592d29e76d78ac',
      'Rodney was looking forward to a date with his girlfriend all day.\nIn the end Rodney got to go on a wonderful date with his girlfriend.\nHe ended up having to work later than expected.\nHe ended up getting off of work earlier than expected.', 'a',
      'The visible endpoints do not establish why later work is preferred to getting off early.'],
    ['172063d25cc1bc24e7de5316084a2438b487156042279e3a4a20ef3ab969de4b',
      'Sandy was really disappointed no one remembered her birthday.\nShe arrived to find all of her friends there for her surprise party.\nSandy came home from a long day sad and ready to exercise.\nSandy came home from a long day sad and ready to go to bed.', 'b',
      'Both alternatives bring Sandy home; the endpoints do not distinguish exercising from sleeping.'],
    ['e472f170b5649e1c02d5b1044a258761ab3dabac57b0811dae90a2e02eef8671',
      'Jesse leans against the plain wooden foot board of her bed.\nShe finishes with a lonely sigh, pining for her recently dead mom.\nJesse folded the worn quilts lying there.\nJesse folded the worn quilts for her mother.', 'a',
      'Both quilting actions can precede remembering her mother; the intended preference needs source adjudication.'],
    ['e2c9a606854c4c0007362b26c90647a0921098ad4b8c0ff00a7874ff641f395b',
      "Emily had an essay due the next day.\nEmily's mother forced her to finish chores.\nEmily stopped washing dishes and started writing.\nEmily started washing dishes an stopped writing.", 'a',
      'Both alternatives can lead to the stated chores; the annotated preference depends on unstated timing.'],
    ['33635093f35a02a840c54dbd73818a740e8398ea276be48a1875e9643ceb4b16',
      'Todd hated making coffee in the morning.\nTodd happily enjoyed his life saving gift.\nTodd forgot to make coffee and was tired driving in to work.\nTodd made coffee anyway to that he would be alert driving in to work.', 'b',
      'The gift and its connection to either hypothesis are missing from the visible story.'],
  ].map(([id, text, annotatedLabel, reason]) => ({ dataset: 'anli', id: id!, aliases: [], text: text!,
    annotatedLabel: annotatedLabel!, reason: reason!, status: 'pending' as const })),
];

/** Stable visible identity also catches legacy batches that omitted dataset_records. */
export function anliReviewText(story: Record<string, unknown>): string | undefined {
  const fields = ['beginning', 'ending', 'a', 'b'].map(key => story[key]);
  return fields.every(value => typeof value === 'string') ? fields.join('\n') : undefined;
}

export function pendingSourceReview(dataset: string, id: string): SourceReview | undefined {
  return SOURCE_REVIEWS.find(review => review.status === 'pending' && review.dataset === dataset &&
    (review.id === id || review.aliases.includes(id)));
}

/** A whole task is held when any of its original source records needs review. */
export function sourceReviewReason(record: Record<string, unknown>):
    string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  const dataset = record.dataset ?? (curriculum?.family === 'anli_batch' ? 'anli' : undefined);
  if (typeof dataset !== 'string') return undefined;
  if (dataset === 'anli') {
    const semantics = record.semantics as { inputs?: { stories?: unknown[] } } | undefined;
    const texts = semantics?.inputs?.stories?.flatMap(story => story && typeof story === 'object' ?
      [anliReviewText(story as Record<string, unknown>)] : []) ?? [];
    if (SOURCE_REVIEWS.some(review => review.dataset === dataset && review.status === 'pending' && texts.includes(review.text)))
      return 'source_review_pending';
  }
  const ids = [record.dataset_records, record.source_ids].flatMap(value => Array.isArray(value) ? value : []);
  return ids.some(id => typeof id === 'string' && pendingSourceReview(dataset, id)) ?
    'source_review_pending' : undefined;
}
