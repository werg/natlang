"""Declarative domains for the synthesizer: a latent world per use case (hidden attributes rendered as short texts)
and the leaf functions that read those attributes back. Registering a domain makes it available to every program
shape and to the compositional synthesizer. Texts are templates on purpose: the teacher re-renders them later."""
from __future__ import annotations

import random

from . import synth as S


def enum_type(values) -> str:
    return " | ".join(f'"{v}"' for v in values)


def register(name: str, arg: str, program_names: list, label: dict, bools: dict, rubric: str = ""):
    """label = {"leaf": canon, "aliases": [...], "type": TypeName, "param": p, "values": {value: [sentences]},
                "instructions": text, "description": text, "locals": [...]}
       bools = {canon: {"aliases": [...], "true": [suffixes], "false": [suffixes], "p": 0.4,
                        "instructions": text, "description": text, "locals": [...]}}"""
    values = list(label["values"])
    param = label["param"]

    def make(rng: random.Random):
        v = rng.choice(values)
        item = {"text": rng.choice(label["values"][v]), label["leaf"]: v}
        for canon, b in bools.items():
            flag = rng.random() < b.get("p", 0.4)
            item["text"] += rng.choice(b["true"] if flag else b["false"])
            item[canon] = flag
        return item

    S.TYPES[label["type"]] = enum_type(values)
    args = {param: "Text", **({"rubric": "Text"} if rubric else {})}
    S.LEAVES[label["leaf"]] = (label["aliases"], args, label["type"], label["instructions"], lambda h, k=label["leaf"]: h[k])
    S.LEAF_KIND[label["leaf"]] = "labels"
    S.LEAF_LOCAL[label["leaf"]] = label["locals"]
    S.DESCRIPTIONS[label["leaf"]] = label["description"]
    S.LEAF_TYPES[label["leaf"]] = [label["type"]]
    if rubric:
        S.LEAF_RUBRIC[label["leaf"]] = rubric
    for canon, b in bools.items():
        S.LEAVES[canon] = (b["aliases"], {param: "Text"}, "Bool", b["instructions"], lambda h, k=canon: h[k])
        S.LEAF_KIND[canon] = "flags"
        S.LEAF_LOCAL[canon] = b["locals"]
        S.DESCRIPTIONS[canon] = b["description"]
    S.DOMAINS[name] = (make, [label["leaf"]] + list(bools))
    S.DOMAIN_ARG[name] = arg
    S.DOMAIN_NAMES[name] = program_names


register(
    "alerts", "alerts", ["alert_digest", "oncall_summary", "incident_overview"],
    {"leaf": "component_of", "aliases": ["component_of", "which_system", "affected_part"], "type": "Component", "param": "alert",
     "locals": ["components", "systems"], "description": "Which system an alert is about.",
     "instructions": "Which system is the alert in `args/alert` about: database, network, storage, or application?",
     "values": {"database": ["Replication lag on db-3 is 40 seconds", "Too many connections on the orders database", "Slow query log is growing fast on db-1"],
                "network": ["Packet loss of 12% between eu-west and eu-central", "The load balancer reports 3 unhealthy backends", "DNS lookups are timing out in the office VLAN"],
                "storage": ["Disk /var on node-7 is 93% full", "The backup volume failed its SMART check", "Object store write latency above 2 s"],
                "application": ["Checkout service returns 500 for 1 in 5 requests", "The mobile API is crash-looping after the last deploy", "Login page renders blank for Safari users"]}},
    {"is_paging": {"aliases": ["is_paging", "needs_a_human_now", "page_someone"], "locals": ["paging", "page_flags"], "p": 0.35,
                   "description": "Does this alert need a human right now?",
                   "instructions": "Does the alert in `args/alert` need a human right now, because customers are affected or data may be lost? Answer true or false.",
                   "true": [" - customers are affected right now.", "; data loss is possible within the hour.", ". Orders are failing as we speak."],
                   "false": [". No customer impact so far.", "; it has been like this for a week.", ". Seen only in staging."]},
     "is_flapping": {"aliases": ["is_flapping", "keeps_recurring"], "locals": ["flapping", "recurring"], "p": 0.3,
                     "description": "Has this alert been going on and off?",
                     "instructions": "Does the alert in `args/alert` say that the problem keeps coming and going? Answer true or false.",
                     "true": [" It has fired and cleared four times today.", " This is the third time this morning."],
                     "false": ["", " First occurrence."]}})

register(
    "clauses", "clauses", ["contract_review", "clause_checklist", "review_agreement"],
    {"leaf": "clause_type", "aliases": ["clause_type", "kind_of_clause", "classify_clause"], "type": "ClauseType", "param": "clause",
     "locals": ["clause_types", "kinds"], "description": "What one contract clause is about.",
     "instructions": "What is the contract clause in `args/clause` about: payment, termination, liability, or confidentiality?",
     "values": {"payment": ["Invoices are payable within 30 days of receipt", "The Customer shall pay a monthly fee of EUR 4,000 in advance", "Late payments bear interest of 8% per year"],
                "termination": ["Either party may terminate this agreement with 90 days written notice", "The Supplier may terminate immediately if fees remain unpaid for 60 days", "This agreement renews for one year unless cancelled"],
                "liability": ["The Supplier's total liability is capped at the fees paid in the prior 12 months", "Neither party is liable for indirect or consequential loss", "The Customer indemnifies the Supplier against third-party claims"],
                "confidentiality": ["Each party keeps the other's confidential information secret for five years", "The Supplier may not name the Customer in marketing", "Confidential information excludes what is already public"]}},
    {"is_onesided": {"aliases": ["is_onesided", "favours_the_supplier", "unbalanced"], "locals": ["onesided", "unbalanced_flags"], "p": 0.4,
                     "description": "Is this clause clearly tilted against the customer?",
                     "instructions": "Is the clause in `args/clause` clearly tilted against the Customer? Answer true or false.",
                     "true": [", and the Customer waives any right to object.", "; the Supplier alone decides whether this condition is met.", ", without any corresponding right for the Customer."],
                     "false": [".", ", and the same applies to the other party.", ", as is usual for agreements of this kind."]}})

register(
    "abstracts", "abstracts", ["screen_studies", "review_screening", "abstract_triage"],
    {"leaf": "study_design", "aliases": ["study_design", "design_of", "kind_of_study"], "type": "Design", "param": "abstract",
     "locals": ["designs", "study_kinds"], "description": "The design of the study in one abstract.",
     "instructions": "What kind of study does the abstract in `args/abstract` describe: trial, cohort, case_report, or review?",
     "values": {"trial": ["We randomised 240 adults to the new inhaler or placebo for 12 weeks", "In this double-blind randomised study, 88 patients received either drug A or drug B"],
                "cohort": ["We followed 5,100 nurses for ten years and recorded sleep habits and heart disease", "Using registry data, we compared outcomes of 12,000 patients by statin use"],
                "case_report": ["We describe a 54-year-old man who developed a rash after starting the drug", "A single patient presented with an unusual combination of symptoms"],
                "review": ["We searched three databases and pooled 17 studies on exercise and depression", "This narrative overview summarises recent work on gut bacteria"]}},
    {"in_adults": {"aliases": ["in_adults", "adult_population"], "locals": ["adult_flags", "adults_only"], "p": 0.6,
                   "description": "Is the study population adult?",
                   "instructions": "Is the study in `args/abstract` about adults (18 or older)? Answer true or false.",
                   "true": [". Participants were aged 18 to 65.", ". All participants were adults.", ""],
                   "false": [". Participants were children aged 6 to 12.", ". The study enrolled adolescents only."]},
     "reports_harm": {"aliases": ["reports_harm", "mentions_side_effects"], "locals": ["harm_flags", "side_effects"], "p": 0.35,
                      "description": "Does the abstract report harms or side effects?",
                      "instructions": "Does the abstract in `args/abstract` report harms or side effects? Answer true or false.",
                      "true": [" Adverse events were more common in the treatment group.", " Two participants withdrew because of side effects."],
                      "false": ["", " Safety outcomes were not reported."]}})

register(
    "emails", "emails", ["inbox_rules", "mail_digest", "sort_my_mail"],
    {"leaf": "email_intent", "aliases": ["email_intent", "what_they_want", "purpose_of"], "type": "Intent", "param": "email",
     "locals": ["intents", "purposes"], "description": "What the sender of one email wants.",
     "instructions": "What does the sender of the email in `args/email` want: meeting, invoice, complaint, or newsletter?",
     "values": {"meeting": ["Could we find half an hour next week to go through the plan", "Are you free for a call about the offer"],
                "invoice": ["Please find attached invoice 2291 for October", "A reminder that invoice 1877 is still open"],
                "complaint": ["I am disappointed with how my request was handled", "This is the second time my order arrived damaged"],
                "newsletter": ["This month in our community: five new tutorials", "Our autumn catalogue is out, see what is new"]}},
    {"has_deadline": {"aliases": ["has_deadline", "mentions_a_date"], "locals": ["deadline_flags", "dated"], "p": 0.4,
                      "description": "Does the email name a date by which something must happen?",
                      "instructions": "Does the email in `args/email` name a date by which something must happen? Answer true or false.",
                      "true": [" Please reply by Friday 14 March.", " We need this settled before the end of the month."],
                      "false": [".", " No rush at all."]},
     "from_landlord": {"aliases": ["from_landlord", "is_from_my_landlord"], "locals": ["landlord_flags"], "p": 0.25,
                       "description": "Is the email from the landlord?",
                       "instructions": "Is the email in `args/email` written by the reader's landlord? Answer true or false.",
                       "true": [" Regards, Mr Petersen (your landlord)", " - your landlord, H. Petersen"],
                       "false": [" Best, Jana", " Kind regards, the team"]}})

register(
    "survey", "answers", ["code_survey", "survey_themes", "feedback_coding"],
    {"leaf": "theme_of", "aliases": ["theme_of", "code_answer", "main_theme"], "type": "Theme", "param": "answer",
     "locals": ["themes", "codes"], "description": "The main theme of one free-text survey answer.",
     "instructions": "What is the survey answer in `args/answer` mainly about: workload, management, pay, or tools?",
     "values": {"workload": ["There is simply too much to do in too little time", "I regularly work through lunch to keep up"],
                "management": ["Decisions are made without asking the people affected", "My manager gives clear direction and listens"],
                "pay": ["My salary has not kept up with rents in this city", "The bonus scheme is fair and transparent"],
                "tools": ["Our laptops are too slow for the software we must run", "The new ticket system saves me an hour a day"]}},
    {"is_negative": {"aliases": ["is_negative", "unhappy_answer"], "locals": ["negative", "unhappy"], "p": 0.55,
                     "description": "Is this answer a complaint?",
                     "instructions": "Is the survey answer in `args/answer` a complaint rather than praise? Answer true or false.",
                     "true": [" It is wearing me down.", " Something has to change."], "false": [" I am happy with it.", " No complaints from me."]}})

register(
    "home", "events", ["home_rules", "night_watch", "house_events"],
    {"leaf": "room_of", "aliases": ["room_of", "where_in_the_house", "location_of"], "type": "Room", "param": "event",
     "locals": ["rooms", "places"], "description": "Where in the house one event happened.",
     "instructions": "Where did the event in `args/event` happen: kitchen, hallway, garage, or bedroom?",
     "values": {"kitchen": ["The smoke sensor above the stove reported a reading of 3", "The fridge door has been open for six minutes"],
                "hallway": ["Motion was detected by the front door", "The hallway light was switched on"],
                "garage": ["The garage door opened", "The car charger stopped drawing power"],
                "bedroom": ["The bedroom window sensor reports open", "The bedroom thermostat was set to 24 degrees"]}},
    {"is_suspicious": {"aliases": ["is_suspicious", "worth_an_alert"], "locals": ["suspicious", "alert_flags"], "p": 0.3,
                       "description": "Should the residents be alerted about this event?",
                       "instructions": "Should the residents be alerted about the event in `args/event`: is it unexpected given what it says about time and presence? Answer true or false.",
                       "true": [" at 03:12 while nobody is home.", " during the holiday, with the house set to away."],
                       "false": [" at 18:40 with the family at home.", " shortly after a resident arrived."]}})

register(
    "applications", "applications", ["screen_applicants", "shortlist", "application_triage"],
    {"leaf": "field_of", "aliases": ["field_of", "background_of", "main_field"], "type": "Field", "param": "application",
     "locals": ["fields", "backgrounds"], "description": "The professional field of one applicant.",
     "instructions": "What is the professional background of the applicant in `args/application`: engineering, design, sales, or finance?",
     "values": {"engineering": ["Eight years building backend services in Go and Python", "Embedded firmware developer for industrial sensors"],
                "design": ["Product designer with a portfolio of mobile banking apps", "Brand and packaging designer, formerly at an agency"],
                "sales": ["Account executive who closed enterprise deals in logistics", "Inside sales lead for a software reseller"],
                "finance": ["Management accountant with experience in manufacturing", "Treasury analyst at a regional bank"]}},
    {"has_led_team": {"aliases": ["has_led_team", "leadership_experience"], "locals": ["leads", "led_flags"], "p": 0.4,
                      "description": "Has the applicant led a team?",
                      "instructions": "Does the application in `args/application` show that the applicant has led a team? Answer true or false.",
                      "true": ["; led a team of six for three years.", "; currently manages four direct reports."],
                      "false": ["; has always worked as an individual contributor.", "."]},
     "can_start_soon": {"aliases": ["can_start_soon", "available_within_a_month"], "locals": ["available", "soon_flags"], "p": 0.5,
                        "description": "Can the applicant start within a month?",
                        "instructions": "Can the applicant in `args/application` start within a month? Answer true or false.",
                        "true": [" Available immediately.", " Notice period: two weeks."], "false": [" Notice period: six months.", " Available from next summer."]}})

register(
    "probe", "readings", ["probe_report", "telemetry_triage", "housekeeping_check"],
    {"leaf": "subsystem_of", "aliases": ["subsystem_of", "which_subsystem"], "type": "Subsystem", "param": "reading",
     "locals": ["subsystems", "parts"], "description": "Which subsystem a telemetry note is about.",
     "instructions": "Which subsystem is the telemetry note in `args/reading` about: power, thermal, comms, or instrument?",
     "values": {"power": ["Battery at 41% and falling 2% per hour", "Solar array current is 0.3 A below the model"],
                "thermal": ["Radiator panel B reads 12 degrees above nominal", "Heater 2 has been on continuously for six hours"],
                "comms": ["Downlink bit error rate has doubled since yesterday", "The last two ground passes were missed"],
                "instrument": ["Spectrometer shutter reports stuck half open", "The camera returns frames with a bright stripe"]}},
    {"needs_action": {"aliases": ["needs_action", "act_before_next_pass"], "locals": ["action_flags", "act_now"], "p": 0.35,
                      "description": "Must something be done before the next ground contact?",
                      "instructions": "Does the telemetry note in `args/reading` describe something that must be dealt with before the next ground contact, because it is getting worse or threatens the mission? Answer true or false.",
                      "true": [" and the trend is accelerating.", "; at this rate safe mode triggers within a day."],
                      "false": ["; stable for the past week.", ", within the expected seasonal range."]}})
