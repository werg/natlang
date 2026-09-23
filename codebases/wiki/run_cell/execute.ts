export default async function execute(block_id: string, input: string): Promise<CellResult> {
return await host.wiki.runCell(block_id, input);
}
