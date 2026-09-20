/** Native operations used by the local studio. Natlang owns their sequencing. */
import { mkdir, writeFile, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { BuildWorkspace } from '../../applications/build_workbench.mjs';
import { MediaWorkspace } from '../../applications/media_workbench.mjs';
import { NotebookWorkspace } from '../../applications/notebook.mjs';
import { PackageRegistry } from '../../applications/package_registry.mjs';
import { RepositoryMigration } from '../../applications/repository_migration.mjs';
const hash = text => createHash('sha256').update(text).digest('hex');
const assert = (test, message) => { if (!test)
    throw new Error(message); };
const packages = [
    { name: 'greetings', version: '1.0.0', description: 'A small greeting function', engines: ['typescript-host'], dependencies: {}, definitions: { main: { args: { name: 'Text' }, returns: 'Text', engine: 'typescript-host', code: 'return "Hello, " + args.name;' } } },
    { name: 'greetings', version: '1.1.0', description: 'A punctuated greeting', engines: ['typescript-host'], dependencies: {}, definitions: { main: { args: { name: 'Text' }, returns: 'Text', engine: 'typescript-host', code: 'return "Hello, " + args.name + "!";' } } },
];
let nextProgress=0;
const progressWaiters=new Map();
async function command(argv, cwd) {
    return new Promise((resolve, reject) => {
        const env = { ...process.env };
        delete env.NODE_TEST_CONTEXT;
        const child = spawn(argv[0], argv.slice(1), { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
        let output = '', output_bytes = 0;
        const capture = (chunk,stream) => {
            output_bytes += chunk.length; output = (output + chunk.toString()).slice(-65536);
            if(process.send){const id=++nextProgress;stream.pause();progressWaiters.set(id,()=>stream.resume());process.send({progress:chunk.toString(),progressId:id},error=>{if(error){progressWaiters.delete(id);stream.resume();}});}
        };
        child.stdout.on('data', chunk=>capture(chunk,child.stdout));
        child.stderr.on('data', chunk=>capture(chunk,child.stderr));
        child.once('error', reject);
        child.once('close', (code, signal) => resolve({ status: code === 0 ? 'succeeded' : 'failed', code, signal, output, output_bytes, truncated: output_bytes > Buffer.byteLength(output) }));
    });
}
export async function studioOperation(operation, payload, root) {
    await mkdir(root, { recursive: true });
    if (operation === 'terminal.run') {
        assert(typeof payload.command === 'string' && payload.command.trim(), 'Command required');
        return command(['bash', '-lc', payload.command], root);
    }
    if (operation === 'notebook.query') {
        const notebook = new NotebookWorkspace([], payload.tables ?? { observations: [{ day: 'Monday', value: 12 }, { day: 'Tuesday', value: 19 }, { day: 'Wednesday', value: 15 }] });
        try {
            return { rows: notebook.query(payload.source) };
        }
        finally {
            notebook.close();
        }
    }
    if (operation === 'build.run') {
        assert(typeof payload.source === 'string' && ['copy', 'uppercase'].includes(payload.operation), 'Invalid build input');
        const id = hash(payload.source + payload.operation), folder = join(root, 'build', id);
        await mkdir(folder, { recursive: true });
        await writeFile(join(folder, 'input.txt'), payload.source);
        const build = await new BuildWorkspace(folder, { cacheDir: join(root, 'build-cache') }).open();
        const result = await build.execute({ id: 'transform', argv: ['@builtin', payload.operation], inputs: ['input.txt'], outputs: ['output.txt'] });
        assert(result.status === 'ok', result.detail);
        return { ...result, output: await readFile(join(folder, 'output.txt'), 'utf8'), events: build.drainEvents() };
    }
    if (operation.startsWith('media.')) {
        const folder = join(root, 'assets');
        await mkdir(folder, { recursive: true });
        const media = await new MediaWorkspace(folder).open();
        const asset = `asset-${randomUUID()}.mp4`;
        if (operation === 'media.sample') {
            const result = await command(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-f', 'lavfi', '-i', 'testsrc2=size=640x360:rate=24', '-t', '4', '-c:v', 'libx264', '-threads', '1', '-pix_fmt', 'yuv420p', join(folder, asset)], folder);
            assert(result.status === 'succeeded', result.output);
            return { asset, probe: await media.probe(asset) };
        }
        assert(operation === 'media.transform', 'Unknown media operation');
        assert(/^asset-[\w-]+\.mp4$/.test(payload.asset), 'Invalid asset ID');
        const plan = { start: 0, end: 2, x: 0, y: 0, width: 320, height: 180, keep_audio: false, ...payload.parameters, kind: payload.kind, input: payload.asset, output: asset };
        const result = await media.render(plan);
        assert(result.status === 'ok', result.detail);
        return { asset, receipt: result, inspection: await media.inspect({ text: 'Studio transformation' }, plan, result), events: media.drainEvents() };
    }
    if (operation.startsWith('packages.')) {
        const registry = new PackageRegistry(packages, join(root, 'packages'));
        if (operation === 'packages.catalog')
            return { catalog: registry.catalog() };
        if (operation === 'packages.resolve')
            return { locks: registry.solutions({ name: payload.name, range: payload.range, engine: 'typescript-host' }) };
        if (operation === 'packages.install')
            return { ...await registry.install(payload.lock, payload.target), events: registry.drainEvents() };
    }
    if (operation === 'repository.check') {
        assert(typeof payload.before === 'string' && typeof payload.after === 'string', 'Source required');
        const folder = join(root, 'candidates', randomUUID());
        await mkdir(folder, { recursive: true });
        await writeFile(join(folder, 'greet.mjs'), payload.before);
        const repository = new RepositoryMigration(folder, { files: ['greet.mjs'], checks: [
                { id: 'syntax', argv: [process.execPath, '--check', 'greet.mjs'] },
                { id: 'behavior', argv: [process.execPath, '--input-type=module', '-e', 'import assert from "node:assert/strict"; import {greet} from "./greet.mjs"; assert.equal(greet("Ada"), "Hello, Ada"); console.log("Greeting contract passed");'] },
            ] });
        const base = await repository.open();
        const candidate = payload.before === payload.after ? base : repository.apply(base.revision, [{ path: 'greet.mjs', old: payload.before, new: payload.after }]);
        const validation = await repository.validate(candidate.revision);
        return { ...repository.report(candidate.revision, validation), events: repository.drainEvents() };
    }
    throw new Error(`Unknown studio operation: ${operation}`);
}
if (process.argv.includes('--worker')) process.on('message',message=>{if(message.progressAck){const resume=progressWaiters.get(message.progressAck);progressWaiters.delete(message.progressAck);resume?.();}});
if (process.argv.includes('--worker'))
    process.once('message', async (request) => {
        try {
            process.send?.({ result: await studioOperation(request.operation, request.payload, request.root) });
        }
        catch (error) {
            process.send?.({ error: String(error) });
        }
        finally {
            process.disconnect();
        }
    });
