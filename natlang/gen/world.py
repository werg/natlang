"""A miniature latent world: hidden records rendered as short texts. The hidden fields are the gold.

Deliberately small and template-based: it exists to prove the pipeline. A teacher model will later
re-render these texts in many voices (SYNTHETIC_DATA.md, Y1), with the hidden record unchanged.
"""
from __future__ import annotations

import random

NAMES = ["Dana Whitfield", "Omar Haddad", "Li Wei", "Priya Raman", "Tomás Ortega", "Ingrid Solberg",
         "Kwame Mensah", "Aiko Tanaka", "Noah Becker", "Lucía Fernández"]
PRODUCTS = ["kettle", "headphones", "desk lamp", "backpack", "router", "blender", "keyboard", "tent"]

TICKETS = {
    "billing": ["I was charged twice for my {month} invoice.", "Please change the card on file, the old one expired.",
                "My refund for the {product} still has not arrived.", "Why is there a {amount} EUR fee on my statement?"],
    "technical": ["The app crashes every time I open the settings page.", "Export to CSV produces an empty file.",
                  "The {product} will not pair with my phone any more.", "Login fails with error 500 since this morning."],
    "spam": ["CHEAP WATCHES!!! Visit our store today for 90% off.", "Congratulations, you won a free cruise, click here.",
             "Boost your followers overnight with our service.", "Hot singles near you want to chat."],
}
URGENT = ["PRODUCTION IS DOWN. Customers cannot log in at all.", "Payments are failing for every customer right now.",
          "Security breach: admin passwords were leaked an hour ago.", "All outgoing email has stopped for everyone."]
CALM = ["Could you update the logo in the footer some time?", "Typo on the About page.",
        "It would be nice to have a dark mode.", "Can the invoice PDF use a bigger font?"]
POSITIVE = ["Love the {product}, works perfectly.", "Great value, would buy the {product} again.",
            "Arrived early and nicely packed. Five stars.", "Exactly what I needed, thank you."]
NEGATIVE = ["The {product} broke after two days. Want my money back.", "Arrived late and the box was crushed.",
            "Nothing like the photo, and support never answered.", "Charged twice and still no refund. Terrible."]
MONTHS = ["January", "March", "June", "October"]
RUBRIC = ("billing: anything about charges, invoices, refunds, or payment methods.\n"
          "technical: anything about the product not working.\n"
          "spam: unsolicited advertising or messages unrelated to our product.\n")


UNCOVERED = {"billing": "charges, refunds or payments", "technical": "the product not working",
             "spam": "unsolicited advertising"}


def _fill(rng, template):
    return template.format(month=rng.choice(MONTHS), product=rng.choice(PRODUCTS), amount=rng.randint(3, 40))


def ticket(rng: random.Random) -> dict:
    cat = rng.choice(list(TICKETS))
    return {"text": _fill(rng, rng.choice(TICKETS[cat])), "category": cat}


def urgency_ticket(rng) -> dict:
    urgent = rng.random() < 0.5
    return {"text": rng.choice(URGENT if urgent else CALM), "urgent": urgent}


def review(rng) -> dict:
    pos = rng.random() < 0.5
    return {"text": _fill(rng, rng.choice(POSITIVE if pos else NEGATIVE)), "positive": pos}


def note(rng, drop=None) -> dict:
    """A call note and its hidden record. `drop` names a required field the note leaves out: then no record
    can be extracted, and `missing` says why."""
    name, order = rng.choice(NAMES), f"{rng.randint(1, 9999):04d}"
    amount = round(rng.uniform(5, 300), 2)
    phone = f"555-{rng.randint(1000, 9999)}" if rng.random() < 0.4 else None
    forms = {None: ["Spoke to {n} about order no. {o} - owed {a:.2f} back.",
                    "{n} called regarding order {o}; refund due: {a:.2f}.",
                    "Refund of {a:.2f} approved for {n} (order {o})."],
             "order_id": ["Spoke to {n} about a recent order - owed {a:.2f} back.",
                          "{n} called; refund due: {a:.2f}. Could not find the order number.",
                          "Refund of {a:.2f} approved for {n}."],
             "amount": ["Spoke to {n} about order no. {o} - a refund is owed.",
                        "{n} called regarding order {o}; refund due, amount to be confirmed.",
                        "Refund approved for {n} (order {o})."]}
    text = rng.choice(forms[drop]).format(n=name, o=order, a=amount)
    if phone:
        text += f" Reach them on {phone}."
    rec = {"customer": name, "order_id": order, "amount": amount}
    if phone:
        rec["phone"] = phone
    missing = {None: "", "order_id": rng.choice(["The note does not give an order id, which the record requires.",
                                                 "`args/note` has no order id; `order_id` cannot be filled in."]),
               "amount": rng.choice(["The note does not say how much the refund is, and `amount` is required.",
                                     "`args/note` gives no refund amount; `amount` cannot be filled in."])}[drop]
    return {"text": text, "record": rec, "missing": missing}


def distinct(rng, make, n):
    """n items with distinct texts, so that a text identifies its hidden record."""
    out, seen = [], set()
    while len(out) < n:
        item = make(rng)
        if item["text"] not in seen:
            seen.add(item["text"])
            out.append(item)
    return out
