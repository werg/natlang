import { crispExamples } from './examples/crisp.mjs';
import { naturalExamples } from './examples/natural.mjs';

/** Self-contained projects available in the browser and the test suite. */
export const examples = [...crispExamples, ...naturalExamples];
export const exampleCategories = ['All', ...new Set(examples.map(example => example.category))];
