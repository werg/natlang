import { http } from 'natlang:services';
export default function respond(id: string, response: Response): boolean {
http.respond(id, response)
return true
}
