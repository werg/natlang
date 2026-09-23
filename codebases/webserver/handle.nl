import apply_submission from "./handle/apply_submission";
import error_response from "./handle/error_response";
import log_request from "./handle/log_request";
import match_route from "./handle/match_route";
import page_content from "./handle/page_content";
import parse_request from "./handle/parse_request";
import respond from "./handle/respond";
import review_submission from "./handle/review_submission";
import session_for from "./handle/session_for";
import static_response from "./handle/static_response";
import submission_page from "./handle/submission_page";
import wrap_page from "./handle/wrap_page";
---
description: Handle one HTTP request against the site state; respond; return the
  new site state. The step of a fold over requests.
args:
  acc: Site
  item: Request
returns: Site
effects:
  - http.respond
---
function handle(acc, item) -> Site

  req     = parse_request(item)                       # exact: query, form fields, session cookie
  session = session_for(acc, req)                     # exact: the visitor's session, new if there is none
  route   = match_route(acc.routes, req)              # exact: which route; not_found / method_not_allowed otherwise
  site    = acc

  if route.kind is "static":
      response = static_response(acc, route)
  else if route.kind is "page":
      content  = page_content(route.purpose, acc.name, acc.about, acc.entries, session)      # the page is written for this request
      response = wrap_page(acc, route, content, session)
  else if route.kind is "form":
      review   = review_submission(route.purpose, req.form)                                    # accept or refuse, with cleaned fields
      site     = apply_submission(acc, review)                                                 # exact: a refused submission changes nothing
      content  = submission_page(route.purpose, review)                                        # thank you, or what to fix
      response = wrap_page(site, route, content, session)
  else:
      response = error_response(route)                 # exact: 404 or 405

  respond(item.id, response)                           # effect: the reply goes out
  return log_request(site, req, session, response)     # exact: remember the session, append to the log
