export const examples = [
  {
    name: 'Structured calculation', root: 'math/summarize.ts',
    files: {
      'math/summarize.ts': `/*---
description: Summarize two numbers with a typed result.
args:
  a: Num
  b: Num
returns: Summary
engine: typescript-host
---*/
return { sum: args.a + args.b, larger: Math.max(args.a, args.b) };
`,
      'math/types.ts': 'type Summary = { sum: Num, larger: Num };\n',
    },
    inputs: { a: 3, b: 5 }, expected: { sum: 8, larger: 5 },
  },
  {
    name: 'Natural instruction', root: 'tasks/answer.nl',
    files: {
      'tasks/answer.nl': `---
description: Produce one typed answer from natural instructions.
returns: Num
---
function answer() -> Num

  Write the number 7 to return.
`,
    },
    inputs: {}, expected: 7,
  },
  {
    name: 'Checked function', root: 'people/greet.ts',
    files: {
      'people/greet.ts': `/*---
description: Greet a person with a checked name.
args:
  person: Person
returns: Text
engine: typescript-host
---*/
return \`Hello, \${args.person.name}!\`;
`,
      'people/types.ts': 'type Person = { name: Text };\n',
    },
    inputs: { person: { name: 'Ada' } }, expected: 'Hello, Ada!',
  },
];
