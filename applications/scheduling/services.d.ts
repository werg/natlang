// Typing of the host service for callable-folder TypeScript (scheduler/enumerate.ts).
declare module 'natlang:services' {
  import type { CalendarService } from './calendar.ts';
  export const calendar: CalendarService;
}
