import { interfaceExamples } from './examples/interfaces.mjs';
import { naturalExamples } from './examples/natural.mjs';

/** The public gallery introduces natlang itself. TypeScript-only programs are test fixtures. */
const byId = id => naturalExamples.find(example => example.id === id);
export const examples = [
  byId('extract-contact'), byId('feedback-tone'), byId('policy-decision'),
  byId('meeting-actions'), byId('evidence-summary'),
  ...interfaceExamples.filter(example => example.modelRequired),
  ...naturalExamples.filter(example => ![
    'natural-seven', 'extract-contact', 'feedback-tone', 'policy-decision',
    'meeting-actions', 'evidence-summary',
  ].includes(example.id)),
];
export const exampleCategories = ['All', ...new Set(examples.map(example => example.category))];
