import { out } from 'natlang:services';
export default function emit_reply(to: string, line: string, action: Action): boolean {
out.emit({ to: to, line: line, action: action.code })
return true
}
