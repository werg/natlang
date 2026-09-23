import { calendar } from 'natlang:services';
export default function add_to_calendar(date: string, email: string): boolean {
calendar.add({ date: date, note: email.slice(0, 80) })
return true
}
