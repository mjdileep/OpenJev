"""Fixed workload matrix for independent Noul, Choice and Score questions."""

from dataclasses import asdict, replace

from openjev import Choice, Noul, Score


def mixed_cases():
    nouls = [
        Noul("Does the customer need action today?"),
        Noul("Is the request about a payment?"),
        Noul("Does the customer report a software error?"),
        Noul("Is the customer asking for a refund?"),
        Noul("Does the customer prefer a phone call?"),
        Noul("Does the customer prefer email?"),
        Noul("Has the issue lasted more than one day?"),
        Noul("Has the customer contacted support before?"),
        Noul("Does the customer mention an invoice?"),
        Noul("Is the customer unable to access their account?"),
        Noul("Does the issue affect other team members?"),
        Noul("Does the customer mention a physical delivery?"),
    ]
    choice_specs = [
        ("Which team should handle this?", ["Billing and payments", "Technical support", "Sales"]),
        ("How should support contact the customer?", ["Email", "Phone", "In-app message"]),
        ("When does the customer need help?", ["Today", "Within a week", "No deadline stated"]),
        (
            "What is the main payment issue?",
            ["Failed payout", "Duplicate charge", "No payment issue"],
        ),
        ("What product is involved?", ["Business account", "Personal account", "Not stated"]),
        (
            "What role does the customer have?",
            ["Account administrator", "Team member", "Not stated"],
        ),
        (
            "Which action is requested?",
            ["Repair the service", "Refund a payment", "Explain pricing"],
        ),
        ("What is the current ticket status?", ["Unresolved", "Resolved", "Not stated"]),
        ("What device is involved?", ["Laptop", "Phone", "Not stated"]),
        ("What is the affected currency?", ["USD", "EUR", "Not stated"]),
        ("How broad is the impact?", ["One person", "Multiple team members", "Not stated"]),
        (
            "Which information should support check first?",
            ["Payment history", "Login logs", "Shipment tracking"],
        ),
    ]
    choices = [
        Choice(question, dict(zip(("a", "b", "c"), options, strict=True)))
        for question, options in choice_specs
    ]
    score_specs = [
        ("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
        ("How urgent is the request?", ["No urgency", "Some urgency", "Immediate urgency"]),
        ("How specific is the description?", ["Vague", "Some details", "Precise details"]),
        ("How serious is the business impact?", ["Minor", "Moderate", "Severe"]),
        (
            "How much financial impact is reported?",
            ["None stated", "Limited impact", "Major impact"],
        ),
        ("How much evidence is provided?", ["No evidence", "Some evidence", "Detailed evidence"]),
        (
            "How much has the customer tried already?",
            ["Nothing stated", "One attempt", "Several attempts"],
        ),
        ("How clear is the requested next action?", ["Unclear", "Partly clear", "Explicit"]),
        (
            "How long has the problem lasted?",
            ["Under one day", "One to three days", "Over three days"],
        ),
        ("How much service is unavailable?", ["No outage", "Partial service", "All service"]),
        (
            "How much follow-up has occurred?",
            ["No follow-up", "One follow-up", "Several follow-ups"],
        ),
        ("How much confidence is justified by the facts?", ["Low", "Medium", "High"]),
    ]
    scores = [Score(question, levels) for question, levels in score_specs]
    message = (
        "I administer our business account. USD payouts have failed for three days. "
        "I retried twice from my laptop and saw error P42. The rest of the account works. "
        "Two other team members have the same issue. We need the money today to pay suppliers. "
        "I contacted support yesterday and the ticket remains unresolved. "
        "This is frustrating. Please fix the payout and email me with an update. "
        "I am not requesting a refund or a phone call."
    )
    notes = "\n".join(
        f"Resolved ticket {1000 + i}: The customer confirmed resolution; "
        "no further action is needed."
        for i in range(28)
    )
    states = {"short": message, "long": notes + "\n\nCurrent customer message:\n" + message}
    profiles = [
        ("balanced-3", (1, 1, 1)),
        ("balanced-12", (4, 4, 4)),
        ("balanced-24", (8, 8, 8)),
        ("noul-heavy", (8, 2, 2)),
        ("choice-heavy", (2, 8, 2)),
        ("score-heavy", (2, 2, 8)),
    ]
    policy = (
        " Use only facts in the current message. Earlier resolved tickets are background. "
        "Keep missing information distinct from explicit denials. "
        "Judge this candidate against this question's criteria."
    )
    explanation = (
        " This description applies only to the current customer's message. "
        "The candidate requires supporting evidence in that message; similar terms in "
        "older resolved tickets do not establish this claim. "
        "Missing details should not be invented when evaluating whether the candidate fits."
    )
    result = []

    def add(profile, counts, context, shape):
        questions = {}
        # Interleave types to exercise batching across question boundaries.
        for i in range(max(counts)):
            for kind, bank, count in zip(
                ("noul", "choice", "score"), (nouls, choices, scores), counts, strict=True
            ):
                if i >= count:
                    continue
                q = bank[i]
                if shape == "long questions":
                    q = replace(q, instructions=q.instructions + policy * 4)
                elif shape == "uneven candidates":
                    if kind == "choice":
                        q = replace(
                            q,
                            criteria={
                                k: v + (explanation * 2 if k == "c" else "")
                                for k, v in q.criteria.items()
                            },
                        )
                    elif kind == "score":
                        q = replace(
                            q,
                            criteria=[
                                v + (explanation if n == 2 else "")
                                for n, v in enumerate(q.criteria)
                            ],
                        )
                elif shape == "wide options":
                    if kind == "choice":
                        q = replace(
                            q,
                            criteria={
                                **q.criteria,
                                "d": "No option fits",
                                "e": "Insufficient evidence",
                            },
                        )
                    elif kind == "score":
                        low, medium, high = q.criteria
                        q = replace(
                            q,
                            criteria=[
                                low,
                                f"Between {low} and {medium}",
                                medium,
                                f"Between {medium} and {high}",
                                high,
                            ],
                        )
                questions[f"{kind}_{i}"] = asdict(q)
        candidate_count = sum(
            1 if q["type"] == "noul" else len(q["criteria"]) for q in questions.values()
        )
        result.append(
            {
                "id": f"{profile}-{context}-{shape.replace(' ', '-')}",
                "group": f"{profile} / {context} / {shape}",
                "profile": profile,
                "context_length": context,
                "suffix_shape": shape,
                "type_counts": dict(zip(("noul", "choice", "score"), counts, strict=True)),
                "question_count": sum(counts),
                "candidate_count": candidate_count,
                "state": states[context],
                "questions": questions,
            }
        )

    for profile, counts in profiles:
        for context in states:
            for shape in ("normal", "long questions", "uneven candidates"):
                add(profile, counts, context, shape)
    for kind, counts in (("noul", (12, 0, 0)), ("choice", (0, 12, 0)), ("score", (0, 0, 12))):
        for context in states:
            add(f"{kind}-only", counts, context, "normal")
    for context in states:
        add("balanced-12", (4, 4, 4), context, "wide options")
    return result
