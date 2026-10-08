/**
 * pi's system prompt sections and resources on the port: section order and tags, the tools and rules built from the
 * selected tools, project context files up the tree, skills discovery, and render determinism.
 * Run: node --test applications/pi/test/prompt-sections.test.mjs
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { piPrompt } from '../extensions/pi-prompt/sections.ts';
import { loadProjectContextFiles, loadSkills } from '../host/resources.ts';

function tree() {
  const root = mkdtempSync(join(tmpdir(), 'pi-prompt-'));
  const agent = join(root, 'agent'), project = join(root, 'repo'), cwd = join(project, 'pkg');
  mkdirSync(join(agent, 'skills', 'deploy'), { recursive: true });
  mkdirSync(join(cwd, '.pi', 'skills'), { recursive: true });
  writeFileSync(join(agent, 'AGENTS.md'), 'global rules');
  writeFileSync(join(project, 'CLAUDE.md'), 'repo rules');
  writeFileSync(join(cwd, 'AGENTS.md'), '﻿pkg rules');
  writeFileSync(join(agent, 'skills', 'deploy', 'SKILL.md'), '---\nname: deploy\ndescription: Ship <it> & check\n---\nbody');
  writeFileSync(join(cwd, '.pi', 'skills', 'notes.md'), '---\ndescription: Take notes\n---\n');
  writeFileSync(join(cwd, '.pi', 'skills', 'nodesc.md'), '---\nname: x\n---\n');
  return { agent, project, cwd };
}

test('context files: the agent directory first, then ancestors from the root down', () => {
  const { agent, project, cwd } = tree();
  const files = loadProjectContextFiles(cwd, agent).filter(file => file.path.startsWith(join(agent, '..')));
  assert.deepEqual(files.map(file => file.content), ['global rules', 'repo rules', 'pkg rules']);
  assert.equal(files[1].path, join(project, 'CLAUDE.md'));
});

test('skills: SKILL.md roots and described .md files; undescribed ones are skipped', () => {
  const { agent, cwd } = tree();
  const skills = loadSkills(cwd, { agentDir: agent });
  assert.deepEqual(skills.map(skill => skill.name), ['deploy', 'skills']);
  assert.equal(skills[1].description, 'Take notes');
});

test('sections: pi order, own tags, tool snippets and rules, deterministic', async () => {
  const { agent, cwd } = tree();
  const extension = piPrompt({ cwd, agentDir: agent, packageDir: '/opt/pi' });
  assert.equal(extension.name, 'pi-prompt');
  assert.deepEqual(extension.sections.map(section => [section.key, section.tag]),
    [['preamble', false], ['tools', false], ['rules', false], ['docs', false], ['project_context', false], ['skills', false], ['cwd', false]]);
  const tool = name => ({ name, description: '', parameters: {} });
  const input = { conversationId: 1, agent: { tools: ['read', 'write', 'edit', 'bash', 'subagent'].map(tool), sections: [], thinkingLevel: 'off', extensions: [] },
    env: { cwd }, shown: {}, read: {} };
  const rendered = Object.fromEntries(await Promise.all(extension.sections.map(async section => [section.key, await section.render(input, {})])));
  assert.match(rendered.preamble, /^You are an expert coding assistant operating inside pi/);
  assert.equal(rendered.tools, '<tools>\n- read: Read file contents\n- write: Create or overwrite files\n- edit: Make precise file edits with exact text replacement, including multiple disjoint edits in one call\n- bash: Execute bash commands (ls, grep, find, etc.)\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>');
  assert.equal(rendered.rules.split('\n')[1], '- Use bash for file operations like ls, rg, find');
  assert.ok(rendered.rules.includes('- You can inspect PI_* environment variables for current model and session details.'));
  assert.ok(rendered.rules.endsWith('- Be concise in your responses\n- Show file paths clearly when working with files\n</rules>'));
  assert.ok(rendered.docs.includes('- Main documentation: /opt/pi/README.md'));
  assert.ok(rendered.project_context.startsWith('<project_context>\nProject-specific instructions and guidelines:\n\n<project_instructions path='));
  assert.ok(rendered.skills.includes('<description>Ship &lt;it&gt; &amp; check</description>'));
  assert.ok(rendered.skills.includes("Use the read tool to load a skill's file"));
  assert.equal(rendered.cwd, `<cwd>\n${cwd}\n</cwd>`);
  const again = Object.fromEntries(await Promise.all(extension.sections.map(async section => [section.key, await section.render({ ...input }, {})])));
  assert.deepEqual(again, rendered);
  const bare = await extension.sections[1].render({ ...input, agent: { ...input.agent, tools: [] } }, {});
  assert.equal(bare, '<tools>\n(none)\n\nIn addition to the tools above, you may have access to other custom tools depending on the project.\n</tools>');
});
