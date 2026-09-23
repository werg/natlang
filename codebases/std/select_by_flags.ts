export default function select_by_flags(items: string[], flags: boolean[]): string[] {
return items.filter((_, i) => flags[i])
}
