import test from 'node:test';
import assert from 'node:assert/strict';
import { validateIterateWorlds } from './authored-iterate-source-builder.mjs';
import { makeGuidedSoftIterateCase } from './semantic-iterate-worlds-v15-soft-guided-builder.mjs';
import { worlds } from './semantic-iterate-worlds-v15-data.mjs';

function withBooleanDecision(source) {
  const field = source.passes[0].allowed_fields[0];
  const world = {
    ...source,
    fields: { ...source.fields },
    initial: { ...source.initial, [field]: 'pending' },
    passes: source.passes.map(pass => ({ ...pass, allowed_fields: [...pass.allowed_fields] })),
    passStates: source.passStates.map(state => ({ ...state, [field]: true })),
    evidence: { ...source.evidence },
    field_enums: Object.fromEntries(Object.entries(source.field_enums ?? {}).filter(([name]) => name !== field)),
    output_types: { [field]: 'boolean' },
    initial_types: { [field]: ['pending'] },
  };
  return { world, field };
}

test('iterate draft contracts preserve mixed boolean and string fields', () => {
  const typedCase = withBooleanDecision(worlds[0]);
  const typed = worlds.map((world, index) => index === 0 ? typedCase.world : world);
  assert.doesNotThrow(() => validateIterateWorlds(typed));
  const row = makeGuidedSoftIterateCase(typed[0], 0, { revision: 'typed-draft-test/1' });
  const code = row.curriculum.reference.root[0][1].code;
  assert.match(code, new RegExp(`type Draft = \\{[^}]*${typedCase.field}: boolean`));
  assert.match(code, new RegExp(`type InitialDraft = \\{[^}]*${typedCase.field}: boolean \\| "pending"`));
  const task = JSON.parse(row.semantics.folder_files['task.json']);
  assert.match(task.output_contract.format, new RegExp(`${typedCase.field} \\(boolean\\)`));
});

test('iterate source validation rejects values that violate a declared primitive type', () => {
  const typedCase = withBooleanDecision(worlds[0]);
  const invalid = worlds.map((world, index) => index === 0 ? typedCase.world : world);
  invalid[0].passStates[3][typedCase.field] = 'true';
  assert.throws(() => validateIterateWorlds(invalid), /outside its declared output type/);
});
