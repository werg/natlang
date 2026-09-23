export default function observe(item: LogEvent): Observation {
return host.logs.observe(item);
}
