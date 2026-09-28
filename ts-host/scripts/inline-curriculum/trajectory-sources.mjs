/** Conservative source trace audit and independently specified file-edit slices.
 * Slices are new scoped tasks with actual source editor actions, not claimed full SWE solutions.
 */
import { sourceCase, safePath, digest, permissive } from './directory-sources.mjs';
import { returnCall } from './lib.mjs';
import { createHash } from 'node:crypto';

const text = value => typeof value === 'string' ? value : Array.isArray(value) ? value.map(v => v.text ?? '').join('\n') : '';
const messages = row => typeof row.messages === 'string' ? JSON.parse(row.messages) : row.messages ?? row.trajectory ?? [];
export function editorPath(value) {
  if (typeof value !== 'string' || !value.startsWith('/testbed/')) throw new Error('unsupported_workspace_root');
  return safePath(value.slice('/testbed/'.length));
}
export function fullView(observation, args) {
  if (args.view_range && (args.view_range[0] !== 1 || args.view_range[1] !== -1)) throw new Error('partial_view');
  if (!observation.includes('cat -n') || /truncat|<response clipped>|omitted|\.\.\. \(\d+ lines/i.test(observation)) throw new Error('untrusted_full_view');
  const lines = observation.replaceAll('\r\n', '\n').split('\n');
  const numbered = lines.map(line => /^\s*(\d+)\t(.*)$/.exec(line)).filter(Boolean);
  if (!numbered.length || numbered.some((m, i) => Number(m[1]) !== i + 1)) throw new Error('noncontiguous_view');
  // cat-n does not preserve final newline status. The fixture explicitly adopts a terminal newline;
  // operation fidelity is independently checked against the source editor replacement/snippet.
  return numbered.map(m => m[2]).join('\n') + '\n';
}

export async function buildTrajectorySources(sources, limit = 8, licenseResolver) {
  const records = [], audits = [], rejected = [], licenseCache = new Map();
  for (const source of ['swesmith', 'nebius', 'nvidia']) {
    if (!sources[source]) continue;
    const { rows, info } = sources[source];
    let added = 0;
    for (const row of rows) {
      const id = row.traj_id ?? row.trajectory_id ?? String(row._source_row_index);
      try {
        if (row.capture_error) throw new Error(row.capture_error);
        if (row.resolved !== true && row.resolved !== 1) throw new Error(row.resolved === -1 ? 'unknown_source_success' : 'source_task_failed');
        const ms = messages(row), calls = [];
        for (const [index, message] of ms.entries()) for (const call of message.tool_calls ?? []) {
          const fn = call.function;
          const args = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : fn.arguments;
          const observations = ms.slice(index + 1).filter(m => m.role === 'tool' &&
            (m.tool_call_id === call.id || m.tool_call_ids?.includes(call.id)));
          const terminal = ['finish', 'submit'].includes(fn.name);
          if (observations.length !== 1 && !(terminal && observations.length === 0)) throw new Error('missing_or_ambiguous_observation');
          calls.push({ name: fn.name, args, observation: text(observations[0]?.content), message_index: index, call_id: call.id });
        }
        const unsupported = [...new Set(calls.filter(c => !['str_replace_editor', 'finish', 'submit'].includes(c.name)).map(c => c.name))];
        audits.push({ source, id, source_success: true, calls: calls.length, unsupported_tools: unsupported,
          full_trajectory: unsupported.length ? 'incompatible_tools' : 'needs_initial_workspace_and_terminal_validation' });
        if (added >= limit) continue;
        const repo = row.repo ?? String(row.instance_id).split('.')[0].replace('__', '/');
        const originalId = row.instance_id;
        const licenseKey = `${repo}:${originalId}`;
        if (!licenseCache.has(licenseKey)) licenseCache.set(licenseKey, await licenseResolver?.(source, row, info, repo));
        const license = licenseCache.get(licenseKey);
        if (!license || !license.pinned || !permissive(license.spdx)) throw new Error('pinned_repository_license_missing');
        const known = new Map();
        let selected;
        for (const call of calls) {
          if (call.name !== 'str_replace_editor') {
            if (!['think', 'finish', 'submit'].includes(call.name)) known.clear(); // opaque command could mutate any file
            continue;
          }
          const { args, observation } = call;
          let path;
          try { path = editorPath(args.path); } catch { known.clear(); continue; }
          if (args.command === 'view') {
            try { known.set(path, { content: fullView(observation, args), call }); } catch { known.delete(path); }
            continue;
          }
          if (args.command !== 'str_replace') { known.delete(path); continue; }
          const state = known.get(path);
          known.delete(path);
          if (!state || typeof args.old_str !== 'string' || !args.old_str || typeof args.new_str !== 'string') continue;
          if (state.content.split(args.old_str).length !== 2 || !observation.includes('has been edited') || /error|no replacement|multiple occurrences/i.test(observation)) continue;
          const post = state.content.replace(args.old_str, args.new_str);
          if (post === state.content || state.content.length + post.length > 16000) continue;
          const displayed = [...observation.replaceAll('\r\n', '\n').matchAll(/^\s*\d+\t(.*)$/gm)].map(m => m[1]);
          if (!displayed.length || !post.includes(displayed.join('\n'))) continue;
          selected = { path, state, post, call }; break;
        }
        if (!selected) {
          // A successful create has a complete payload and an absent-path precondition. It can be
          // made an independent explicit-content task without retaining any opaque earlier commands.
          const create = calls.find(call => call.name === 'str_replace_editor' && call.args.command === 'create' &&
            typeof call.args.file_text === 'string' && call.args.file_text.length > 0 && call.args.file_text.length < 6000 &&
            call.observation.includes(`File created successfully at: ${call.args.path}`));
          if (!create) throw new Error('no_independent_verified_editor_slice');
          const path = editorPath(create.args.path);
          if (path === 'creation-request.json') throw new Error('request_path_collision');
          const payload = create.args.file_text;
          const files = { 'creation-request.json': JSON.stringify({ path, content: payload }, null, 2) + '\n' };
          const record = sourceCase({ source, info, sourceId: `${id}:create:${create.message_index}`, group: `swe:repo:${repo}`,
            task: 'Create the file specified in creation-request.json with exactly its supplied content. The target does not yet exist. Preserve the request file and return the target path. This task checks file creation; do not execute the supplied code.',
            files, expectedFiles: { ...files, [path]: payload }, expected: path,
            actions: [['read_file', { path: 'creation-request.json' }], ['write_file', { path, content: payload }], returnCall(path)],
            license: `${info.license} trajectory; ${license.spdx} repository`,
            adaptation: 'independent-source-file-creation; explicit-content-task; absent-target precondition; original payload unchanged' });
          record.gold_sources = ['source-editor-success-observation', 'exact-native-scoped-creation-oracle'];
          record.source_groups.push(`repository:${repo}`);
          record.external_source.trajectory = { dataset: info.dataset, id, original_instance_id: originalId,
            original_row_sha256: digest(row), metadata_revision: info.metadata_revision,
            source_success: true, conversion_scope: 'independent_file_creation',
            source_message_indices: [create.message_index], source_calls: [create],
            native_wrapper_steps: ['read explicit creation request', 'typed terminal return'],
            repository: repo, repository_license: license, whole_issue_replayed: false };
          records.push(record); added++;
          continue;
        }
        const { path, state, post, call } = selected;
        const request = { path, find: call.args.old_str, replace_with: call.args.new_str };
        if (path === 'change-request.json') throw new Error('request_path_collision');
        const files = { [path]: state.content, 'change-request.json': JSON.stringify(request, null, 2) + '\n' };
        const record = sourceCase({ source, info, sourceId: `${id}:edit:${call.message_index}`, group: `swe:repo:${repo}`,
          task: 'Apply the exact correction in change-request.json to its target file. Read the file before editing it. Preserve every other file and return the target path. This is a scoped file correction, not the complete repository issue.',
          files, expectedFiles: { ...files, [path]: post }, expected: path,
          actions: [['read_file', { path: 'change-request.json' }], ['read_file', { path }], ['edit_file', request], returnCall(path)],
          license: `${info.license} trajectory; ${license.spdx} repository`,
          adaptation: 'independent-editor-slice; explicit edit specification; reconstructed full view with canonical final newline; source reasoning not imported' });
        record.gold_sources = ['source-editor-success-observation', 'exact-native-scoped-edit-oracle'];
        record.source_groups.push(`repository:${repo}`);
        record.external_source.trajectory = { dataset: info.dataset, id, original_instance_id: originalId,
          original_row_sha256: digest(row), metadata_revision: info.metadata_revision,
          source_success: true, conversion_scope: 'independent_editor_slice',
          source_message_indices: [state.call.message_index, call.message_index],
          source_calls: [state.call, call], repository: repo, repository_license: license,
          native_wrapper_steps: ['read explicit edit request', 'typed terminal return'],
          original_task: ms.filter(m => m.role === 'user').map(m => text(m.content)).join('\n'),
          whole_issue_replayed: false };
        records.push(record); added++;
      } catch (error) { rejected.push({ source, id, reason: error.message }); }
    }
  }
  return { records, audits, rejected };
}

/** SWE-smith instance IDs supply the original repository commit. Fetch only license text, never execute. */
export async function pinnedRepositoryLicense(source, row, info, repo) {
  if (source !== 'swesmith') return null; // others require their corresponding task/base-commit join
  if (!/^[\w.-]+\/[\w.-]+$/.test(repo)) return null;
  const revision = String(row.instance_id).split('.')[1];
  const current = info.repository_licenses?.[repo];
  if (!/^[a-f0-9]{8,40}$/.test(revision ?? '') || !permissive(current?.spdx)) return null;
  const filename = current.url.split('/').at(-1);
  const url = `https://raw.githubusercontent.com/${repo}/${revision}/${filename}`;
  const response = await fetch(url);
  if (!response.ok) return null;
  const bytes = Buffer.from(await response.arrayBuffer()), body = bytes.toString('utf8');
  const gitBlob = createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
  if (gitBlob !== current.license_blob_sha) return null; // require the classified terms to match at the pinned commit
  const pattern = current.spdx === 'MIT' ? /Permission is hereby granted[\s\S]*THE SOFTWARE IS PROVIDED/i :
    current.spdx === 'Apache-2.0' ? /Apache License[\s\S]*Version 2\.0/i : /Redistribution and use in source and binary forms/i;
  if (!pattern.test(body)) return null;
  return { spdx: current.spdx, pinned: true, revision, url, sha256: digest(body), text: body };
}
