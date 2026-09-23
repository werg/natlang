export default function search(observation: Observation): Evidence[] {
return host.logs.query(observation);
}
