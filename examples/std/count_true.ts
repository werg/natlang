export default function count_true(flags: boolean[]): number {
return flags.filter(Boolean).length
}
