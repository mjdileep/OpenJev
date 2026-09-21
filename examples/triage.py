from openjev import Choice, DecisionEngine, Noul, Score

with DecisionEngine.from_pretrained() as engine:
    result = engine.decide(
        state="Help! My payouts have been failing for 3 days.",
        questions={
            "urgent": Noul("Does the message convey urgency?"),
            "department": Choice(
                "Which team should handle this?",
                {"billing": "Payments and refunds", "technical": "Bugs and integrations"},
            ),
            "frustration": Score(
                "How frustrated is the customer?", ["Calm", "Frustrated", "Angry"]
            ),
        },
    )
    print(result.to_dict())
