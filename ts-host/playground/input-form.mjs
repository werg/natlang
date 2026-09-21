import { TypeEnv, formatType } from '../dist/browser/natlang.js';

const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);

function accepts(value, type, env) {
  const resolved = env.resolve(type);
  switch (resolved.kind) {
    case 'prim': return resolved.name === 'Null' ? value === null :
      resolved.name === 'Num' ? typeof value === 'number' && Number.isFinite(value) &&
        (!Number.isInteger(value) || Number.isSafeInteger(value)) :
        resolved.name === 'Bool' ? typeof value === 'boolean' : typeof value === 'string';
    case 'lit': return value === resolved.value;
    case 'union': return resolved.members.some(member => accepts(value, member, env));
    case 'list': return Array.isArray(value) && value.every(item => accepts(item, resolved.element, env));
    case 'dict': return object(value) && Object.entries(value).every(([key, item]) =>
      !key.startsWith('$') && accepts(item, resolved.element, env));
    case 'record': return object(value) && Object.keys(value).every(key =>
      !key.startsWith('$') && resolved.fields.some(field => field.name === key)) && resolved.fields.every(field =>
      field.optional && !(field.name in value) || field.name in value && accepts(value[field.name], field.type, env));
    default: return false;
  }
}

export function mountInputForm(container, lambda, values, onChange) {
  const env = new TypeEnv(lambda.types);
  const fields = lambda.type.params.fields;
  const bindings = [];
  container.replaceChildren();

  const build = (field, current, path) => {
    const type = env.resolve(field.type);
    const label = path.join('.');
    const nested = type.kind === 'record';
    const wrap = document.createElement(nested ? 'fieldset' : 'label');
    wrap.className = nested ? 'input-record' : 'input-field';
    const heading = document.createElement(nested ? 'legend' : 'span');
    const name = document.createElement('span'); name.textContent = field.name;
    const hint = document.createElement('code'); hint.textContent = formatType(field.type);
    heading.append(name, hint); wrap.append(heading);

    let control, read;
    const literals = type.kind === 'union' && type.members.every(member => env.resolve(member).kind === 'lit') ?
      type.members.map(member => env.resolve(member).value) : null;
    if (nested) {
      const children = type.fields.map(child => build(child, object(current) ? current[child.name] : undefined,
        [...path, child.name]));
      const group = document.createElement('div'); group.className = 'input-record-fields';
      group.append(...children.map(child => child.wrap)); wrap.append(group);
      read = () => Object.fromEntries(children.map(child => [child.field.name, child.read()])
        .filter(([, value]) => value !== undefined));
    } else if (literals) {
      control = document.createElement('select');
      control.append(...literals.map(value => new Option(String(value), JSON.stringify(value))));
      control.value = JSON.stringify(current ?? literals[0]);
      read = () => JSON.parse(control.value);
    } else if (type.kind === 'prim' && type.name === 'Bool') {
      control = document.createElement('input'); control.type = 'checkbox'; control.checked = Boolean(current);
      read = () => control.checked;
    } else if (type.kind === 'prim' && type.name === 'Num') {
      control = document.createElement('input'); control.type = 'number'; control.step = 'any';
      control.value = current === undefined ? '' : String(current);
      read = () => {
        if (control.value.trim() === '' || !Number.isFinite(Number(control.value))) throw new Error(`${label} must be a number`);
        return Number(control.value);
      };
    } else if (type.kind === 'prim' && type.name === 'Text') {
      control = current?.length > 100 || String(current ?? '').includes('\n') ?
        document.createElement('textarea') : document.createElement('input');
      if (control.tagName === 'INPUT') control.type = 'text';
      control.value = current ?? '';
      read = () => control.value;
    } else {
      control = document.createElement('textarea'); control.className = 'json-input';
      control.value = JSON.stringify(current ?? (type.kind === 'list' ? [] : type.kind === 'record' || type.kind === 'dict' ? {} : null), null, 2);
      read = () => { try { return JSON.parse(control.value); } catch { throw new Error(`${label} must be valid JSON`); } };
    }
    if (control) { control.setAttribute('aria-label', label); wrap.append(control); }
    if (field.optional) {
      const include = document.createElement('input'); include.type = 'checkbox'; include.checked = current !== undefined;
      const toggle = document.createElement('span'); toggle.className = 'input-optional';
      toggle.append(include, document.createTextNode('Include'));
      wrap.append(toggle);
      if (control) control.disabled = !include.checked;
      include.addEventListener('change', () => { if (control) control.disabled = !include.checked; onChange(); });
      const baseRead = read;
      read = () => include.checked ? baseRead() : undefined;
    }
    control?.addEventListener('change', onChange);
    return { field, wrap, read, label };
  };

  if (!fields.length) {
    const empty = document.createElement('p'); empty.className = 'input-empty';
    empty.textContent = 'This function has no inputs.'; container.append(empty);
  } else {
    bindings.push(...fields.map(field => build(field, values[field.name], [field.name])));
    container.append(...bindings.map(binding => binding.wrap));
  }

  const validate = value => {
    if (!object(value)) throw new Error('Inputs must be an object');
    for (const { field, wrap, label } of bindings) {
      const valid = field.optional && !(field.name in value) ||
        field.name in value && accepts(value[field.name], field.type, env);
      wrap.classList.toggle('invalid', !valid);
      if (!valid) throw new Error(`${label} must match ${formatType(field.type)}`);
    }
    const names = new Set(fields.map(field => field.name));
    const extra = Object.keys(value).find(name => !names.has(name));
    if (extra) throw new Error(`${extra} is not an input to this function`);
    return value;
  };
  const read = () => {
    const value = {};
    for (const { field, read: readField } of bindings) {
      const item = readField();
      if (item !== undefined) value[field.name] = item;
    }
    return validate(value);
  };
  return { read, validate };
}
