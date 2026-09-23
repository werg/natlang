export default async function probe(input: string): Promise<Clip> {
return await host.media.probe(input);
}
