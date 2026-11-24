import { interfaceExamples } from './examples/interfaces.mjs';
import { naturalExamples } from './examples/natural.mjs';

/** The public gallery introduces natlang itself in Semantic-FP progression. TypeScript-only programs are test fixtures. */
const byId = id => naturalExamples.find(example => example.id === id);
const featured = [
  'inbox-planner',
  'calibrated-gate',
  'support-triage',
  'pluggable-policy',
  'policy-lenses',
  'release-review',
  'semantic-join-rank',
  'policy-compiler',
  'soft-summary',
  'extract-contact',
  'feedback-tone',
  'policy-decision',
  'meeting-actions',
  'evidence-summary',
];
export const examples = [
  ...featured.map(byId).filter(Boolean),
  ...interfaceExamples.filter(example => example.modelRequired),
  ...naturalExamples.filter(example => example.id !== 'natural-seven' && !featured.includes(example.id)),
];
export const exampleCategories = ['All', ...new Set(examples.map(example => example.category))];
