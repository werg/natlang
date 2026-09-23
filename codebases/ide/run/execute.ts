export default async function execute(input: string, revision: string): Promise<RunReport> {
return await host.ide.run({ input: input }, revision);
}
