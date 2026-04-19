import json
import os
from sentence_transformers import SentenceTransformer, util
import pandas as pd
from tabulate import tabulate

class PipelineEvaluator:
    def __init__(self, ground_truth_path="ground_truth_old.json"):
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)
        
    def evaluate(self, detected_conflicts):
        results = []
        total_tp = 0
        total_fp = 0
        total_fn = 0

        # Flatten ground truth for easier comparison
        gt_inconsistencies = []
        for story in self.ground_truth['stories']:
            for inc in story['inconsistencies']:
                gt_inconsistencies.append({
                    "story_id": story['story_id'],
                    "type": inc['conflict_type'],
                    "entity": inc['entity_involved'],
                    "s1": inc['source_sentence_1'],
                    "s2": inc['source_sentence_2'],
                    "matched": False
                })

        # Track pipeline hits
        pipeline_conflicts = []
        for c in detected_conflicts:
            pipeline_conflicts.append({
                "type": c['type'],
                "char": c['details'].get('char', 'Unknown'),
                "sources": c['details'].get('sources', []),
                "matched": False
            })

        # Calculate TPs and FNs
        print("\n--- Matching Pipeline Results against Ground Truth ---")
        for gt in gt_inconsistencies:
            found_match = False
            best_sim = 0
            
            for pc in pipeline_conflicts:
                if pc['matched']: continue
                
                # Check semantic similarity between source sentences
                if pc['sources']:
                    # Combine GT sentences and PC sources for comparison
                    gt_combined = gt['s1'] + " " + gt['s2']
                    pc_combined = " ".join(pc['sources'])
                    
                    sim = util.cos_sim(
                        self.similarity_model.encode(gt_combined), 
                        self.similarity_model.encode(pc_combined)
                    ).item()
                    
                    if sim > 0.6: # Threshold for semantic match
                        found_match = True
                        pc['matched'] = True
                        gt['matched'] = True
                        total_tp += 1
                        results.append({
                            "Status": "✅ TRUE POSITIVE",
                            "GT Type": gt['type'],
                            "Found Type": pc['type'],
                            "Entity": gt['entity'],
                            "Similarity": f"{sim:.2f}"
                        })
                        break
            
            if not found_match:
                total_fn += 1
                results.append({
                    "Status": "❌ FALSE NEGATIVE (Missed)",
                    "GT Type": gt['type'],
                    "Found Type": "-",
                    "Entity": gt['entity'],
                    "Similarity": "0.00"
                })

        # Calculate FPs
        for pc in pipeline_conflicts:
            if not pc['matched']:
                total_fp += 1
                results.append({
                    "Status": "⚠️ FALSE POSITIVE (Noise)",
                    "GT Type": "-",
                    "Found Type": pc['type'],
                    "Entity": pc['char'],
                    "Similarity": "-"
                })

        # Metrics
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(tabulate(pd.DataFrame(results), headers='keys', tablefmt='grid'))
        
        print("\n--- Evaluation Metrics ---")
        print(f"True Positives:  {total_tp}")
        print(f"False Positives: {total_fp} (Inflation)")
        print(f"False Negatives: {total_fn}")
        print(f"Precision:       {precision:.2f}")
        print(f"Recall:          {recall:.2f}")
        print(f"F1 Score:        {f1:.2f}")

        return {"precision": precision, "recall": recall, "f1": f1}

if __name__ == "__main__":
    # Example usage: this would be called by orchestrator or main
    pass
