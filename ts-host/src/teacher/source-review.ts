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
];

export function pendingSourceReview(dataset: string, id: string): SourceReview | undefined {
  return SOURCE_REVIEWS.find(review => review.status === 'pending' && review.dataset === dataset &&
    (review.id === id || review.aliases.includes(id)));
}

/** A whole task is held when any of its original source records needs review. */
export function sourceReviewReason(record: Record<string, unknown>):
    string | undefined {
  const dataset = record.dataset;
  if (typeof dataset !== 'string') return undefined;
  const ids = [record.dataset_records, record.source_ids].flatMap(value => Array.isArray(value) ? value : []);
  return ids.some(id => typeof id === 'string' && pendingSourceReview(dataset, id)) ?
    'source_review_pending' : undefined;
}
