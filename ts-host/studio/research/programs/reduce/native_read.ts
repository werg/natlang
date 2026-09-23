import { research } from 'natlang:services';
export default async function native_read(id: string, offset: number, length: number): Promise<string> {
return await research.nativeRead(id, offset, length);
}
