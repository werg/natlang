export default async function native_read(id: string, offset: number, length: number): Promise<string> {
return await host.research.nativeRead(id, offset, length);
}
