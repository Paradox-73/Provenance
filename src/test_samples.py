from logic_baseline import LogicValidator

def run_tests():
    validator = LogicValidator()
    
    # Test Data: Random, normal scenarios
    test_cases = [
        # --- CONTRADICTION CASES ---
        {
            "type": "Contradiction",
            "premise": "The customized table was made entirely of mahogany wood.",
            "hypothesis": "The table was constructed out of pure steel."
        },
        {
            "type": "Contradiction",
            "premise": "Sarah turned off the lights and went to sleep at 10 PM.",
            "hypothesis": "Sarah was reading a book under the bright lamp at midnight."
        },

        # --- ENTAILMENT CASES (Logical Agreement) ---
        {
            "type": "Entailment",
            "premise": "The astronaut successfully landed on Mars.",
            "hypothesis": "The astronaut has traveled to another planet."
        },
        {
            "type": "Entailment",
            "premise": "My dog chased the mailman down the street.",
            "hypothesis": "An animal was running on the street."
        },

        # --- NEUTRAL CASES (Unrelated Information) ---
        {
            "type": "Neutral",
            "premise": "The CEO announced a merger with the competitor.",
            "hypothesis": "The CEO prefers drinking coffee over tea."
        },
        {
            "type": "Neutral",
            "premise": "It is raining heavily outside today.",
            "hypothesis": "My brother just bought a new laptop."
        }
    ]

    print(f"\n{'TYPE':<15} | {'VERDICT':<15} | {'CONFIDENCE':<10} | {'RESULT'}")
    print("-" * 60)

    for case in test_cases:
        verdict, scores = validator.check_friction(case['premise'], case['hypothesis'])
        
        # Check if the model got it right
        result = "PASS" if verdict == case['type'] else "FAIL"
        
        # Get the confidence score of the predicted label
        confidence = f"{scores[verdict]}%"
        
        print(f"{case['type']:<15} | {verdict:<15} | {confidence:<10} | {result}")

if __name__ == "__main__":
    run_tests()