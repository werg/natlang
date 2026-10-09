"""Deterministic ticket corpus for the batched-execution gates (plans/BATCHED_EXECUTION.md section 8).

    python3 scripts/bench_batched_gen.py OUTDIR   # writes OUTDIR/tickets-1000.json and tickets-12.json
"""
import json
import pathlib
import random
import sys

r = random.Random(7)
billing = ["I was charged twice for invoice #{n} this month and need a refund.",
           "My card ending {n} was declined but the payment page says paid.",
           "Please resend invoice #{n}; the PDF is blank.",
           "The annual plan price changed from $90 to $120 without notice (order {n})."]
tech = ["The export job fails with error E{n} every time I click Download.",
        "Login returns a 500 error for all users in our team since {n} minutes ago.",
        "The mobile app crashes on startup after update 4.{n}.",
        "API responses are taking over {n} seconds and customers are timing out."]
spam = ["Buy cheap watches now!!! Limit {n} pieces, click here.",
        "You won a free cruise #{n}. Reply with your bank details.",
        "Boost your SEO ranking in {n} days, guaranteed."]
other = ["Could you add dark mode? Not urgent (idea {n}).",
         "How do I change the avatar on my profile? (ref {n})",
         "Security alert: someone accessed account {n} from an unknown country and moved funds."]
pool = billing + tech + spam   # every ticket is covered by the default rubric
tickets = [r.choice(pool).format(n=r.randint(100, 99999)) for _ in range(1000)]
out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
(out / "tickets-1000.json").write_text(json.dumps(tickets))
(out / "tickets-12.json").write_text(json.dumps(tickets[:12]))
