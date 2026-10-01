import { Folder, type FolderSnapshot } from '../native/scoped-fs.js';
import type { ProgramContract } from './types.js';
/** Pin complete sources. The authored router receives deployment inputs and candidate names only. */
export type PortfolioSignature = { parameters: string; arguments: string[]; returns: string };
export function composePortfolio(members: readonly { name: string; source: FolderSnapshot; contract: ProgramContract }[], signature: PortfolioSignature, fallback = members[0]?.name): FolderSnapshot {
  if(!members.some(member=>member.name===fallback))throw Error('fallback must name a portfolio member');
  const inputSignature = signature.parameters, inputNames = signature.arguments;
  if (!signature.returns || inputNames.some(name => !/^[A-Za-z_$][\w$]*$/.test(name))) throw new Error('a declared external return type and argument names are required');
  if (!members.length || members.some(member => !/^[A-Za-z_$][\w$]*$/.test(member.name)) || new Set(members.map(member => member.name)).size !== members.length) throw new Error('distinct identifier names are required');
  const files: Record<string, string> = {};
  for (const member of members) for (const path of member.source.filePaths()) files[`members/${member.name}/${path}`] = new TextDecoder('utf-8', { fatal: true }).decode(member.source.readBytesSync(path));
  files['route.nl'] = `---\nargs:\n  inputs: unknown[]\n  choices: string[]\nreturns: string\n---\nChoose the best portfolio member for these deployment inputs from choices. Use only the inputs and member names, never labels, expected outputs, split membership or evaluation evidence. Return exactly one supplied name.\n`;
  files['main.ts'] = `import route from './route.nl';\n${members.map(member => member.contract.exportName === 'default' ? `import ${member.name} from './members/${member.name}/${member.contract.entry.replace(/\.ts$/, '')}';` : `import { ${member.contract.exportName} as ${member.name} } from './members/${member.name}/${member.contract.entry.replace(/\.ts$/, '')}';`).join('\n')}\nexport async function solve(${inputSignature}): Promise<${signature.returns}> {\n  const choices = ${JSON.stringify(members.map(member => member.name))};\n  let selected = ${JSON.stringify(fallback)};\n  try { selected = await route([${inputNames.join(', ')}], choices); } catch { /* A semantic routing failure uses the declared fallback. */ }\n  ${members.map(member => `if (selected === ${JSON.stringify(member.name)}) return ${member.name}(${inputNames.join(', ')});`).join('\n  ')}\n  return ${fallback}(${inputNames.join(', ')});\n}\n`;
  files['portfolio.json'] = JSON.stringify({ schema:'natlang.portfolio/1',fallback, members:members.map(member=>({name:member.name,source:member.source.digest,contract:member.contract})) });
  return Folder.fromFiles(files).snapshot();
}
