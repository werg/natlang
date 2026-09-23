export default async function diff(left: string, right: string): Promise<string> {
return await host.research.diff(left, right);
}
