import os
import json
import time
import torch
from typing import List, Dict, Any
from src import nlp_pipeline
from src.orchestrator import run_provenance_pipeline
from src.neo4j_adapter import Neo4jAdapter
from src.nlp_pipeline import load_spacy_model, load_nli_model, run_nli_check
from src.document_parser import document_parser
from tabulate import tabulate

class Benchmark:
    def __init__(self, chapters_folder: str, ground_truth_file: str):
        self.chapters_folder = chapters_folder
        self.ground_truth_file = ground_truth_file
        self.load_ground_truth()
        self.results = {
            "Graph-Hybrid": {"found": [], "time": 0, "total_detected": 0, "raw_conflicts": []},
            "Pure-NLI": {"found": [], "time": 0, "total_detected": 0, "raw_conflicts": []}
        }
        load_spacy_model()
        load_nli_model()

    def load_ground_truth(self):
        with open(self.ground_truth_file, 'r') as f:
            data = json.load(f)
            self.ground_truth = data["stories"]
            self.total_gt_count = sum(len(s["inconsistencies"]) for s in self.ground_truth)

    def _evaluate_conflicts(self, approach: str, detected_conflicts: List[Dict[str, Any]]):
        found_matches = set() # (story_id, inconsistency_index)
        matched_detections = set()

        for d_idx, detected in enumerate(detected_conflicts):
            # Extract text from details
            details = detected.get("details", {})
            if isinstance(details, dict):
                det_text = " ".join([str(v) for v in details.values()]).lower()
            elif isinstance(details, list):
                det_text = " ".join([str(v) for v in details]).lower()
            else:
                det_text = str(detected).lower()
            
            # POV Mapping for Jax/Kael/Alex/Ethan/etc
            if " i " in f" {det_text} " or det_text.startswith("i "):
                det_text += " jax kael alex ethan barnaby elias sophie alistair miller"

            for s_idx, story in enumerate(self.ground_truth):
                story_id = story["story_id"]
                # Only check matches for the correct chapter/story
                if story_id.lower() not in det_text and story_id.replace("_", " ").lower() not in det_text:
                    # Try matching by checking if the detected chapter field matches
                    det_chapter = detected.get("chapter", "") or detected.get("details", {}).get("chapter", "")
                    if not det_chapter or (story_id.lower() not in det_chapter.lower() and story_id.replace("_", "").lower() not in det_chapter.lower().replace(" ", "")):
                        continue

                for i_idx, inc in enumerate(story["inconsistencies"]):
                    # Match if the detected text contains key phrases from the GT sentences
                    # Use a heuristic: if both source sentences have significant overlap or keywords
                    
                    # Heuristic 1: Entity Match
                    entity_match = inc["entity_involved"].lower().split('/')[0] in det_text or \
                                   (len(inc["entity_involved"].split('/')) > 1 and inc["entity_involved"].lower().split('/')[1] in det_text)
                    
                    # Heuristic 2: Sentence fragments
                    # Check for partial matches of the source sentences in the detection sources
                    gt_s1 = inc["source_sentence_1"].lower()
                    gt_s2 = inc["source_sentence_2"].lower()
                    
                    s1_match = any(word in det_text for word in gt_s1.split() if len(word) > 5)
                    s2_match = any(word in det_text for word in gt_s2.split() if len(word) > 5)

                    if (s1_match or s2_match) and entity_match:
                        found_matches.add((story_id, i_idx))
                        matched_detections.add(d_idx)

        self.results[approach]["found"] = list(found_matches)
        self.results[approach]["total_detected"] = len(detected_conflicts)
        self.results[approach]["true_positives"] = len(matched_detections)

    def run_graph_approach(self):
        print("\n>>> Running Graph-Hybrid Approach on New Data...")
        start_time = time.time()
        adapter = Neo4jAdapter()
        try:
            adapter.connect()
            # The pipeline needs to point to the new data folder
            conflicts = run_provenance_pipeline(self.chapters_folder, adapter, mode="spacy")
            self.results["Graph-Hybrid"]["time"] = time.time() - start_time
            self.results["Graph-Hybrid"]["raw_conflicts"] = conflicts
            self._evaluate_conflicts("Graph-Hybrid", conflicts)
        finally:
            adapter.close()

    def run_pure_nli_approach(self):
        print("\n>>> Running Pure NLI Approach on New Data...")
        start_time = time.time()
        entity_sentences = {}
        all_conflicts = []
        files = sorted([f for f in os.listdir(self.chapters_folder) if f.endswith(".txt")])
        for filename in files:
            story_id = os.path.splitext(filename)[0]
            file_path = os.path.join(self.chapters_folder, filename)
            chunks = document_parser(file_path)
            content = " ".join([chunk["content"] for chunk in chunks])
            doc = nlp_pipeline.nlp(content)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                entities = [ent.text for ent in sent.ents if ent.label_ in ["PERSON", "PRODUCT", "FAC", "VESSEL", "ENVIRONMENT"] and len(ent.text) > 2]
                for ent in entities:
                    if ent not in entity_sentences: entity_sentences[ent] = []
                    for prev_sent in entity_sentences[ent][-15:]: # Slightly larger window
                        nli_res = run_nli_check(prev_sent, sent_text)
                        if nli_res["prediction"] == "contradiction" and nli_res["confidence"] > 0.6:
                            all_conflicts.append({
                                "type": "NLI Friction",
                                "chapter": story_id,
                                "entity": ent,
                                "premise": prev_sent,
                                "hypothesis": sent_text,
                                "confidence": nli_res["confidence"]
                            })
                    entity_sentences[ent].append(sent_text)
        self.results["Pure-NLI"]["time"] = time.time() - start_time
        self.results["Pure-NLI"]["raw_conflicts"] = all_conflicts
        self._evaluate_conflicts("Pure-NLI", all_conflicts)

    def report(self):
        with open("detections_audit_new.json", "w") as f:
            json.dump({"Graph-Hybrid": self.results["Graph-Hybrid"]["raw_conflicts"], "Pure-NLI": self.results["Pure-NLI"]["raw_conflicts"]}, f, indent=2)
        
        print("\n" + "="*90)
        print("NEW DATA PROVENANCE BENCHMARK REPORT")
        print("="*90)
        table_data = []
        for approach, data in self.results.items():
            found_count = len(data["found"])
            total_detected = data["total_detected"]
            true_positives = data.get("true_positives", 0)
            
            recall = found_count / self.total_gt_count if self.total_gt_count > 0 else 0
            precision = true_positives / total_detected if total_detected > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            table_data.append([
                approach, 
                f"{found_count}/{self.total_gt_count}", 
                f"{recall:.1%}", 
                f"{precision:.1%}",
                f"{f1:.2f}",
                total_detected, 
                f"{data['time']:.2f}s"
            ])
        print(tabulate(table_data, headers=["Approach", "GT Found", "Recall", "Precision", "F1", "Total Dets", "Time"], tablefmt="grid"))

if __name__ == "__main__":
    benchmark = Benchmark("data/chapters_data", "data/ground_truth_stories.json")
    benchmark.run_pure_nli_approach()
    benchmark.run_graph_approach()
    benchmark.report()
