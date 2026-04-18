import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class LogicValidator:
    def __init__(self, model_name="cross-encoder/nli-deberta-v3-base"):
        """
        Initializes the DeBERTa-v3 model for Natural Language Inference.
        """
        print(f"Loading NLI model: {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"Model loaded successfully on {self.device}.")

    def check_friction(self, premise, hypothesis):
        """
        Compares two narrative assertions to detect logical friction.
        """
        features = self.tokenizer(
            premise, 
            hypothesis, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            scores = self.model(**features).logits
            
            # Convert logits to probabilities
            probs = torch.nn.functional.softmax(scores, dim=1)[0]
            
            # Typical NLI labels: 0: Contradiction, 1: Entailment, 2: Neutral
            # Note: Label mapping can vary slightly by model, but this is standard for cross-encoders
            labels = ["Contradiction", "Entailment", "Neutral"]
            
            results = {label: round(prob.item() * 100, 2) for label, prob in zip(labels, probs)}
            
            predicted_index = torch.argmax(probs).item()
            prediction = labels[predicted_index]

        return prediction, results

if __name__ == "__main__":
    validator = LogicValidator()

    # Test Case 1: Checking narrative logic
    # Premise: The established rule or previous action.
    # Hypothesis: The new action we are auditing.
    
    established_context = "I went to the party and played poker."
    new_draft_line = "I lost in blackjack at the party."

    print("\n--- Running Friction Check ---")
    print(f"Context (Graph State): {established_context}")
    print(f"New Assertion: {new_draft_line}")
    
    verdict, confidence_scores = validator.check_friction(established_context, new_draft_line)
    
    print(f"\nVerdict: {verdict}")
    print(f"Confidence Matrix: {confidence_scores}")