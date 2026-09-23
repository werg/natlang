import type { Label, Report } from "../../types.js";
export default function is_short(text: string): boolean {
return wordCount(text) <= 60
}
