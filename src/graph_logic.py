# OPEN: src/graph_logic.py
# PASTE the entire file content below:

from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check

# Looks for one person in two different locations WITHIN THE SAME CHAPTER
IMPOSSIBLE_LOCATION_QUERY = """
MATCH (p:Person)-[r1:LOCATED_AT]->(l1)
MATCH (p)-[r2:LOCATED_AT]->(l2)
WHERE l1 <> l2 
AND r1.chapter = r2.chapter
RETURN p.name as character_name, 
       l1.name as location1, 
       l2.name as location2, 
       r1.chapter as conflict_time, 
       r1.text as source1, 
       r2.text as source2
"""

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter

    def _format_inconsistency(self, conflict_type: str, details: dict, confidence: float) -> dict:
        return {
            "type": conflict_type,
            "description": f"Detected {conflict_type} inconsistency.",
            "details": details,
            "confidence": confidence
        }

    def detect_temporal_inconsistency(self) -> list[dict]:
        print("Detecting temporal/impossible location inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IMPOSSIBLE_LOCATION_QUERY)
        inconsistencies = []
        
        seen_pairs = set()
        
        # STOP LIST: Ignore these common false positive "locations"
        IGNORED_LOCATIONS = ["pm", "am", "pocket", "hand", "mind", "table", "chair", "floor"]

        for record in results:
            loc1 = record['location1'].lower()
            loc2 = record['location2'].lower()
            
            # 1. heuristic Filter: Skip if location is in the stop list
            if any(x in loc1 for x in IGNORED_LOCATIONS) or any(x in loc2 for x in IGNORED_LOCATIONS):
                continue

            # Deduplicate (A-B is same as B-A)
            locs = sorted([record['location1'], record['location2']])
            key = f"{record['character_name']}_{locs[0]}_{locs[1]}"
            if key in seen_pairs: continue
            seen_pairs.add(key)

            # 2. NLI Verification (The "Brain")
            # We ask the AI: "Does 'X is at A' contradict 'X is at B'?"
            premise = f"{record['character_name']} was at {record['location1']}."
            hypothesis = f"{record['character_name']} was at {record['location2']}."
            
            print(f"   Analyzing: '{premise}' vs '{hypothesis}'...")
            nli_result = run_nli_check(premise, hypothesis)
            
            # Only report if the AI is confident it's a contradiction
            if nli_result["prediction"] == "contradiction" and nli_result["confidence"] > 0.5:
                inconsistencies.append(self._format_inconsistency(
                    "Temporal/Impossible Location",
                    {
                        "entity": record['character_name'],
                        "location_A": record['location1'],
                        "source_A": record['source1'],
                        "location_B": record['location2'],
                        "source_B": record['source2'],
                        "scope": record['conflict_time']
                    },
                    confidence=nli_result["confidence"]
                ))
                
        return inconsistencies

    # Placeholders for other types
    def detect_inventory_inconsistency(self): return []
    def detect_location_inconsistency(self): return []
    def detect_identity_inconsistency(self): return []
    def detect_attribute_inconsistency(self): return []
    
    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        return all_conflicts