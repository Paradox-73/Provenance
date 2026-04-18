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

# --- FULL GROUND TRUTH DATA ---
GROUND_TRUTH = [
    {"chapter": "Chapter 1", "type": "Temporal Location Conflict", "entity": "Kael", "description": "Bridge vs Engineering at 08:00"},
    {"chapter": "Chapter 1", "type": "Attribute Conflict", "entity": "Kael", "description": "Jagged scar vs smooth face"},
    {"chapter": "Chapter 1", "type": "Inventory Conflict", "entity": "Kael", "description": "Possesses vs Lost Red Onyx Key"},
    {"chapter": "Chapter 1", "type": "Identity Conflict", "entity": "Doctor", "description": "Dr. Aris vs Dr. Evans"},
    {"chapter": "Chapter 1", "type": "Time Logic Inconsistency", "entity": "Kael", "description": "Yesterday vs one hour ago"},
    {"chapter": "Chapter 2", "type": "Temporal Location Conflict", "entity": "Maria", "description": "Quarters vs Hangar Bay at 10:00"},
    {"chapter": "Chapter 2", "type": "Physical State Inconsistency", "entity": "Water", "description": "Liquid vs Frozen solid"},
    {"chapter": "Chapter 3", "type": "Identity Conflict", "entity": "Aris", "description": "Medical Doctor vs Chief Engineer"},
    {"chapter": "Chapter 3", "type": "Naming/Identity Confusion", "entity": "Aris", "description": "Aris vs Evans names"},
    {"chapter": "Chapter 3", "type": "Object Mutation", "entity": "Pod", "description": "Stasis pod replaced by fuel cell"},
    {"chapter": "Chapter 3", "type": "Identity Conflict", "entity": "Maria", "description": "Captain vs Stowaway"},
    {"chapter": "Chapter 3", "type": "Attribute Transfer", "entity": "Aris", "description": "Kael's scar appears on Aris"},
    {"chapter": "Chapter 4", "type": "Attribute Conflict", "entity": "Key", "description": "Marble-sized/Smooth vs Foot-long/Jagged"},
    {"chapter": "Chapter 4", "type": "Environment", "entity": "Light", "description": "Amber vs Violet light"},
    {"chapter": "Chapter 4", "type": "Cyclical Object State", "entity": "Key", "description": "Shard reverts to Stone"},
    {"chapter": "Chapter 4", "type": "Vocal Attribute Shift", "entity": "Kael", "description": "Normal vs Grinding metal voice"},
    {"chapter": "Chapter 5", "type": "Inventory Conflict", "entity": "Kael", "description": "Holding Key vs dropped in core"},
    {"chapter": "Chapter 5", "type": "Spatial Paradox", "entity": "Bridge", "description": "Bridge vs Reactor Core environment"},
    {"chapter": "Chapter 5", "type": "Physical Intangibility", "entity": "Kael", "description": "Solid vs Ghost-like"},
    {"chapter": "Chapter 6", "type": "Temporal Location Conflict", "entity": "Maria", "description": "Bridge vs Engine Room at 12:00"},
    {"chapter": "Chapter 6", "type": "Attribute Conflict", "entity": "Kael", "description": "Blue vs Green eyes"},
    {"chapter": "Chapter 6", "type": "Ship Identity Conflict", "entity": "Starlight", "description": "Starlight vs Void Voyager"}
]

class Benchmark:
    def __init__(self, input_folder: str):
        self.input_folder = input_folder
        self.results = {
            "Graph-Hybrid": {"found": [], "time": 0, "total_detected": 0, "raw_conflicts": []},
            "Pure-NLI": {"found": [], "time": 0, "total_detected": 0, "raw_conflicts": []}
        }
        load_spacy_model()
        load_nli_model()

    def _evaluate_conflicts(self, approach: str, detected_conflicts: List[Dict[str, Any]]):
        found_indices = set()
        for detected in detected_conflicts:
            det_text = str(detected).lower()
            for i, gt in enumerate(GROUND_TRUTH):
                # Heuristic matching
                chapter_match = gt["chapter"].lower() in det_text
                entity_match = gt["entity"].lower() in det_text
                type_words = gt["type"].lower().split()
                type_match = any(word in det_text for word in type_words if len(word) > 3)
                
                if chapter_match and entity_match and type_match:
                    found_indices.add(i)
        
        self.results[approach]["found"] = list(found_indices)
        self.results[approach]["total_detected"] = len(detected_conflicts)

    def run_graph_approach(self):
        print("\n>>> Running Graph-Hybrid Approach...")
        start_time = time.time()
        adapter = Neo4jAdapter()
        try:
            adapter.connect()
            conflicts = run_provenance_pipeline(self.input_folder, adapter, mode="spacy")
            self.results["Graph-Hybrid"]["time"] = time.time() - start_time
            self.results["Graph-Hybrid"]["raw_conflicts"] = conflicts
            self._evaluate_conflicts("Graph-Hybrid", conflicts)
        finally:
            adapter.close()

    def run_pure_nli_approach(self):
        print("\n>>> Running Pure NLI Approach...")
        start_time = time.time()
        entity_sentences = {}
        all_conflicts = []
        files = sorted([f for f in os.listdir(self.input_folder) if f.endswith(".txt")])
        for filename in files:
            chapter_name = os.path.splitext(filename)[0]
            file_path = os.path.join(self.input_folder, filename)
            chunks = document_parser(file_path)
            content = " ".join([chunk["content"] for chunk in chunks])
            doc = nlp_pipeline.nlp(content)
            for sent in doc.sents:
                sent_text = sent.text.strip()
                entities = [ent.text for ent in sent.ents if ent.label_ in ["PERSON", "PRODUCT", "FAC", "VESSEL", "ENVIRONMENT"]]
                for ent in entities:
                    if ent not in entity_sentences: entity_sentences[ent] = []
                    for prev_sent in entity_sentences[ent][-10:]:
                        nli_res = run_nli_check(prev_sent, sent_text)
                        if nli_res["prediction"] == "contradiction" and nli_res["confidence"] > 0.6:
                            all_conflicts.append({
                                "type": "NLI Friction",
                                "chapter": chapter_name,
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
        # Save for manual audit
        with open("detections_audit.json", "w") as f:
            json.dump({"Graph-Hybrid": self.results["Graph-Hybrid"]["raw_conflicts"], "Pure-NLI": self.results["Pure-NLI"]["raw_conflicts"]}, f, indent=2)
        
        print("\n" + "="*60)
        print("COMPREHENSIVE PROVENANCE BENCHMARK REPORT")
        print("="*60)
        table_data = []
        for approach, data in self.results.items():
            recall = len(data["found"]) / len(GROUND_TRUTH)
            table_data.append([approach, f"{len(data['found'])}/{len(GROUND_TRUTH)}", f"{recall:.1%}", len(data["raw_conflicts"]), f"{data['time']:.2f}s"])
        print(tabulate(table_data, headers=["Approach", "GT Found", "Recall", "Total Detections", "Time"], tablefmt="grid"))

if __name__ == "__main__":
    benchmark = Benchmark("chapters_data")
    benchmark.run_pure_nli_approach()
    benchmark.run_graph_approach()
    benchmark.report()
