// Renders every page in src/ to build/, with the shared footer from data/footer.txt.
import { mkdirSync, readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs';
if (existsSync('build')) { console.error('build/ exists; delete it first'); process.exit(1); }
mkdirSync('build');
const footer = readFileSync('data/footer.txt', 'utf8').trim();
for (const page of readdirSync('src')) {
  const body = readFileSync(`src/${page}`, 'utf8').trim();
  writeFileSync(`build/${page.replace(/\.md$/, '.html')}`, `<main>${body}</main>\n<footer>${footer}</footer>\n`);
}
console.log('built', readdirSync('build').length, 'pages');
