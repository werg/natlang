import { readFileSync } from 'node:fs';
import { createContext, runInContext, runInThisContext } from 'node:vm';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { WorkspaceModules, findPackageWorkspace, packageDeclaration } from './workspace-modules.js';
import { packageNameFromSpecifier } from './package-specifier.js';
import { markRaces } from './runtime/context.js';
import { EvalFailure, consoleWriter, withinTimeout } from './native/evaluator.js';
export { EvalFailure } from './native/evaluator.js';
const ARGUMENTS_KEY = '__natlang_arguments';
const prelude = readFileSync(fileURLToPath(new URL('../prelude.js', import.meta.url)), 'utf8');
function snapshot(value) {
    // Eval scope follows TypeScript: an absent optional property may be undefined.
    // Published portable results still use the stricter portable() boundary below.
    if (value === undefined || value === null || typeof value === 'string' || typeof value === 'boolean')
        return value;
    if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value)))
        return value;
    if (Array.isArray(value))
        return Object.freeze(Array.from(value, snapshot));
    if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
        const out = Object.create(null);
        for (const [key, item] of Object.entries(value))
            out[key] = snapshot(item);
        return Object.freeze(out);
    }
    throw new TypeError('eval scope contains a nonportable value');
}
export function portable(value, seen = new Set(), path = '$') {
    if (value === null || typeof value === 'string' || typeof value === 'boolean')
        return value;
    if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value)))
        return value;
    if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || value instanceof ArrayBuffer)
        throw new TypeError(`eval result ${path} contains an unsupported or inexact ${typeof value} value`);
    if (typeof value.then === 'function')
        throw new TypeError('await the asynchronous imported function call');
    if (seen.has(value))
        throw new TypeError('eval result contains a cycle');
    seen.add(value);
    if (Array.isArray(value)) {
        const out = Array.from(value, (item, index) => portable(item, seen, `${path}[${index}]`));
        seen.delete(value);
        return out;
    }
    if (Object.prototype.toString.call(value) !== '[object Object]')
        throw new TypeError('eval result contains a native object');
    const proto = Object.getPrototypeOf(value);
    if (proto !== null && (Object.getPrototypeOf(proto) !== null || proto.constructor?.name !== 'Object'))
        throw new TypeError('eval result contains a native class instance');
    const out = Object.create(null);
    for (const [key, item] of Object.entries(value))
        out[key] = portable(item, seen, `${path}.${key}`);
    seen.delete(value);
    return out;
}
function lowerStaticImports(source, validateImport) {
    const file = ts.createSourceFile('imports.ts', source, ts.ScriptTarget.Latest, true);
    const edits = [];
    const visit = (node) => {
        if (ts.isImportDeclaration(node)) {
            if (!ts.isStringLiteral(node.moduleSpecifier))
                throw new Error('Package import must use a string literal');
            validateImport(node.moduleSpecifier.text);
            const clause = node.importClause;
            const call = `await __natlang_import(${node.moduleSpecifier.getText(file)})`;
            const fields = [];
            let text = '';
            const typeOnlyBindings = clause?.namedBindings && ts.isNamedImports(clause.namedBindings)
                && clause.namedBindings.elements.length > 0 && clause.namedBindings.elements.every(binding => binding.isTypeOnly);
            if (!clause?.isTypeOnly && !(typeOnlyBindings && !clause?.name)) {
                if (clause?.name)
                    fields.push(`default: ${clause.name.text}`);
                const bindings = clause?.namedBindings;
                if (bindings && ts.isNamedImports(bindings))
                    for (const binding of bindings.elements)
                        if (!binding.isTypeOnly)
                            fields.push(`${binding.propertyName?.text ?? binding.name.text}: ${binding.name.text}`);
                if (fields.length)
                    text += `const { ${fields.join(', ')} } = ${call};`;
                if (bindings && ts.isNamespaceImport(bindings))
                    text += `const ${bindings.name.text} = ${call};`;
                if (!text)
                    text = `${call};`;
            }
            edits.push({ start: node.getStart(file), end: node.end, text });
        }
        else
            ts.forEachChild(node, visit);
    };
    visit(file);
    for (const edit of edits.sort((a, b) => b.start - a.start))
        source = source.slice(0, edit.start) + edit.text + source.slice(edit.end);
    return source;
}
function compile(code, body, asyncBody = false, modules = false, validateImport = () => { }) {
    // The body is a function expression; the scope, live values, and import hook arrive as its parameters,
    // so nothing of the runtime's plumbing is a global that eval code could find on globalThis.
    let source = body ? `(${asyncBody ? 'async ' : ''}function (self: any, __live: any, __natlang_import: any) {\n${code}\n})` +
        `(...(globalThis as any)[${JSON.stringify(ARGUMENTS_KEY)}])` : code;
    // Reparse lowered imports before TS binding; otherwise TS rewrites uses to removed import aliases.
    if (modules)
        source = lowerStaticImports(source, validateImport);
    const result = ts.transpileModule(source, {
        fileName: 'natlang-eval.ts', reportDiagnostics: true,
        compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
            isolatedModules: true, removeComments: false },
        transformers: modules ? { before: [context => root => {
                    const visit = node => {
                        if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword)
                            return ts.factory.updateCallExpression(node, ts.factory.createIdentifier('__natlang_import'), undefined, node.arguments.map(arg => ts.visitNode(arg, visit)));
                        return ts.visitEachChild(node, visit, context);
                    };
                    return ts.visitNode(root, visit);
                }] } : undefined,
    });
    const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
    if (errors.length)
        throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
    return result.outputText;
}
/** Raw native bindings and process-ending calls that model-written eval code must not reach: a malformed call into
 * process.binding('fs') aborted the whole collector natively (2026-10-07, 96 workers lost). Everything else on
 * process (env, cwd, argv, hrtime, ...) stays as it was. */
const HIDDEN_PROCESS_MEMBERS = new Set(['binding', '_linkedBinding', 'dlopen', 'abort', 'reallyExit', '_kill',
    'kill', 'exit', 'chdir', 'setuid', 'setgid', 'seteuid', 'setegid', 'setgroups', 'initgroups']);
const evalProcess = new Proxy(process, {
    get(target, key, receiver) {
        if (typeof key === 'string' && HIDDEN_PROCESS_MEMBERS.has(key))
            return () => { throw new Error(`process.${key} is not available to evaluated code`); };
        return Reflect.get(target, key, receiver);
    },
});
/**
 * Node evaluator: a `vm` context in this process. Portable scope data arrives as a frozen snapshot
 * (`self`); live objects, callables, captures and the output sink arrive by reference (`__live`).
 */
/**
 * Rejections nothing handles of promises eval code made: its realm's Promise is theirs. Model code can drop a promise
 * (an async callback given to forEach) whose rejection would end the host as an unhandled rejection; it is recorded
 * as a host event of the eval's environment instead. Any other unhandled rejection is the host's own and, with no
 * other listener, still ends the process as it would without this one.
 */
const evalRealms = new WeakMap();
let watchingRejections = false;
function watchEvalRejections(realmPromise, report) {
    evalRealms.set(realmPromise, report);
    if (watchingRejections)
        return;
    watchingRejections = true;
    process.on('unhandledRejection', function evalRejection(reason, promise) {
        const owner = evalRealms.get(promise.constructor ?? {});
        if (owner)
            return owner(reason);
        if (process.listenerCount('unhandledRejection') === 1)
            throw reason;
    });
}
export class TypeScriptEnvironment {
    authority = 'shared-node-host';
    mode;
    /** Synchronous run time allowed per eval; unlimited unless set. */
    timeoutMs;
    scopeCapabilities;
    modules;
    workspace;
    packageEvents = [];
    context;
    disposed = false;
    observe;
    /** Cancellers of timers that eval code scheduled and that have not run yet. */
    timers = new Set();
    constructor(options = {}) {
        this.mode = options.mode ?? 'fresh';
        this.timeoutMs = options.timeoutMs;
        this.observe = options.observe;
        this.workspace = options.workspace ?? findPackageWorkspace(process.cwd());
        this.scopeCapabilities = { allowModules: true, allowNetwork: options.network ?? true };
        if (this.workspace)
            this.modules = new WorkspaceModules(this.workspace, event => this.packageEvents.push(event));
        if (this.timeoutMs !== undefined && (!Number.isInteger(this.timeoutMs) || this.timeoutMs < 1))
            throw new RangeError('timeoutMs must be positive');
    }
    fork() {
        return new TypeScriptEnvironment({ mode: 'fresh',
            timeoutMs: this.timeoutMs, observe: this.observe, workspace: this.workspace, network: this.scopeCapabilities.allowNetwork });
    }
    makeContext() {
        // Node's host globals are available to eval code as they are to any module in the application.
        // A callback that eval code schedules may throw after its eval has finished; that is recorded as a host
        // event instead of becoming an uncaught exception that ends the application.
        const guarded = (callback) => typeof callback !== 'function' ? callback : (...args) => {
            const report = (error) => this.packageEvents.push({ operation: 'eval.callback-error',
                message: error instanceof Error ? error.message : String(error) });
            try {
                const value = callback(...args);
                if (value && typeof value.catch === 'function')
                    value.catch(report);
            }
            catch (error) {
                report(error);
            }
        };
        // Timers belong to the call: close() clears those still pending, so nothing eval code scheduled outlives it.
        // setInterval is not offered; repeated work is an iterateOn loop.
        const timers = this.timers, cancellers = new WeakMap();
        const scheduled = (start, stop, callback) => {
            const run = guarded(callback);
            const handle = start(() => { timers.delete(cancel); run(); });
            const cancel = () => stop(handle);
            timers.add(cancel);
            cancellers.set(handle, cancel);
            return handle;
        };
        const cancelling = (stop) => (handle) => {
            const cancel = handle && typeof handle === 'object' ? cancellers.get(handle) : undefined;
            if (cancel) {
                timers.delete(cancel);
                cancel();
            }
            else
                stop(handle);
        };
        const context = createContext({ console: undefined, process: evalProcess, Buffer,
            clearTimeout: cancelling(clearTimeout), clearImmediate: cancelling(clearImmediate),
            setTimeout: (callback, ...rest) => scheduled(run => setTimeout(run, ...rest), clearTimeout, callback),
            setImmediate: (callback) => scheduled(run => setImmediate(run), clearImmediate, callback),
            queueMicrotask: (callback) => queueMicrotask(guarded(callback)),
            structuredClone, performance, crypto: globalThis.crypto,
            TextEncoder, TextDecoder, URL, URLSearchParams, AbortController, AbortSignal, Blob });
        runInContext(prelude, context);
        const realmPromise = runInContext('Promise', context);
        markRaces(realmPromise);
        watchEvalRejections(realmPromise, reason => this.packageEvents.push({
            operation: 'eval.unhandled-rejection', message: reason instanceof Error ? reason.message : String(reason)
        }));
        if (this.scopeCapabilities.allowNetwork)
            Object.assign(context, {
                fetch: async (input, init) => {
                    const url = new URL(input instanceof Request ? input.url : String(input));
                    const event = { operation: 'network.fetch', url: url.origin + url.pathname,
                        method: init?.method ?? (input instanceof Request ? input.method : 'GET'), replayable: false };
                    try {
                        const response = await globalThis.fetch(input, init);
                        this.packageEvents.push({ ...event, status: response.status });
                        return response;
                    }
                    catch (error) {
                        this.packageEvents.push({ ...event, status: 'failed' });
                        throw error;
                    }
                }, URL, URLSearchParams, Headers, Request, Response, AbortController, AbortSignal,
            });
        return context;
    }
    captureConsole(context) {
        const logs = [];
        const write = consoleWriter(logs, value => { try {
            return JSON.stringify(portable(value));
        }
        catch {
            return String(value);
        } });
        context.console = Object.freeze({ log: write, info: write, warn: write, error: write });
        return logs;
    }
    declarationOf(name) {
        return this.modules && this.scopeCapabilities.allowModules ? packageDeclaration(this.modules.workspace, name) : undefined;
    }
    importModule = async (specifier) => {
        if (!this.modules)
            throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
        return this.modules.load(specifier);
    };
    prepare(request) {
        if (this.disposed)
            throw new Error('TypeScript environment is disposed');
        if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
            throw new TypeError('invalid eval request');
        const context = this.mode === 'retained' ? (this.context ??= this.makeContext()) : this.makeContext();
        const logs = this.captureConsole(context);
        const scope = snapshot(request.scope);
        // The body picks its arguments up as it starts; the property is not enumerable and is removed right after.
        Object.defineProperty(context, ARGUMENTS_KEY, { value: [scope, request.live ?? {}, this.importModule],
            enumerable: false, configurable: true });
        return { context, logs, start: (code, timeout) => {
                try {
                    return runInContext(code, context, { ...(timeout === undefined ? {} : { timeout }), displayErrors: true });
                }
                finally {
                    delete context[ARGUMENTS_KEY];
                }
            } };
    }
    failure(request, error, logs) {
        return new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'), { sourceStack: error && typeof error === 'object' && 'stack' in error && typeof error.stack === 'string' ?
                error.stack : undefined, logs });
    }
    execute(request) {
        let logs = [];
        const timeout = request.timeoutMs ?? this.timeoutMs;
        try {
            const prepared = this.prepare(request);
            logs = prepared.logs;
            const code = '"use strict";\n' + compile(request.code, request.body);
            const value = prepared.start(code, timeout);
            if (value && typeof value === 'object' && typeof value.then === 'function')
                throw new TypeError('an asynchronous result needs executeAsync');
            return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
        }
        catch (error) {
            throw this.failure(request, error, logs);
        }
    }
    async executeAsync(request) {
        let logs = [];
        const timeout = request.timeoutMs ?? this.timeoutMs;
        try {
            const prepared = this.prepare(request);
            logs = prepared.logs;
            const code = '"use strict";\n' + compile(request.code, request.body, request.body, true, specifier => {
                packageNameFromSpecifier(specifier);
                if (!this.modules)
                    throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
            });
            const value = await withinTimeout(Promise.resolve(prepared.start(code, timeout)), timeout);
            return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
        }
        catch (error) {
            this.clearTimers();
            throw this.failure(request, error, logs);
        }
    }
    /**
     * Evaluate a callable-folder module body in the host realm, so its values interoperate with
     * application code (arrays, classes, Promises). Bindings are passed by reference.
     */
    evaluateModule(code, bindings) {
        if (this.disposed)
            throw new Error('TypeScript environment is disposed');
        const names = Object.keys(bindings);
        const factory = runInThisContext(`(function (${names.join(', ')}) {\n"use strict";\n${code}\n})`, { displayErrors: true });
        return factory(...names.map(name => bindings[name]));
    }
    capture(request, status) {
        const events = this.packageEvents.splice(0);
        for (const event of events)
            this.observe?.(event);
        const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body, status };
        this.observe?.(evalEvent);
        return [...events, evalEvent];
    }
    /** Cancel the timers eval code scheduled that have not run: when the call ends, or when an eval fails. */
    clearTimers() {
        if (!this.timers.size)
            return;
        this.observe?.({ operation: 'eval.timers-cleared', count: this.timers.size });
        for (const cancel of this.timers)
            cancel();
        this.timers.clear();
    }
    close() { this.disposed = true; this.context = undefined; this.clearTimers(); }
}
