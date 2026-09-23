export default function trace(run_id: string, index: number): TraceView {
return host.ide.inspect(run_id, index);
}
