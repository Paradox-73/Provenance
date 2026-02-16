from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check

# 1. TEMPORAL QUERY
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

# 2. INVENTORY QUERY (Generic)
# We now fetch ALL assertions and filter them semantically in Python
INVENTORY_CONFLICT_QUERY = """
MATCH (p:Person)-[r1:HAS_ASSERTION]->(item:Entity)
MATCH (p)-[r2:HAS_ASSERTION]->(item)
WHERE r1.chapter = r2.chapter
AND r1.id <> r2.id
RETURN p.name as character_name,
       item.name as item_name,
       r1.predicate as pred_A,
       r2.predicate as pred_B,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter
        # Cache for verb classification to improve speed
        self.verb_cache = {}

    def _format_inconsistency(self, conflict_type: str, details: dict, confidence: float) -> dict:
        return {
            "type": conflict_type,
            "description": f"Detected {conflict_type} inconsistency.",
            "details": details,
            "confidence": confidence
        }

    def _classify_predicate(self, entity, predicate, item) -> str:
        """
        Uses NLI to semantically classify a predicate into a category.
        Returns: 'POSSESSION', 'LOSS', or 'OTHER'
        """
        cache_key = predicate.lower()
        if cache_key in self.verb_cache:
            return self.verb_cache[cache_key]

        # We construct a hypothesis for each category
        # Context: "The person [predicate] the item."
        premise = f"{entity} {predicate} the {item}."
        
        # Test for Possession
        possession_hyp = f"{entity} possesses the {item}."
        pos_result = run_nli_check(premise, possession_hyp)
        
        # Test for Loss
        loss_hyp = f"{entity} does not possess the {item}."
        loss_result = run_nli_check(premise, loss_hyp)

        classification = "OTHER"
        if pos_result["prediction"] == "entailment" and pos_result["confidence"] > 0.5:
            classification = "POSSESSION"
        elif loss_result["prediction"] == "entailment" and loss_result["confidence"] > 0.5:
            classification = "LOSS"
        
        # Heuristic override for clear "loss" verbs if NLI is fuzzy
        if any(v in predicate for v in ["lost", "dropped", "missing"]):
            classification = "LOSS"

        self.verb_cache[cache_key] = classification
        return classification

    # --- MODULE 1: TEMPORAL ---
    def detect_temporal_inconsistency(self) -> list[dict]:
        print("Detecting temporal/impossible location inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IMPOSSIBLE_LOCATION_QUERY)
        inconsistencies = []
        seen_pairs = set()
        
        # Hardcoded stopwords are still standard practice for Named Entity noise
        IGNORED_LOCATIONS = [
            "pm", "am", "pocket", "hand", "mind", "table", "chair", "floor", "console",
            "room", "corridor", "hallway", "side", "front", "back",
            "she", "he", "him", "her", "it", "they", "them", "me", "you", 
            "hours", "minutes", "seconds", "moment", "instant", "time", "day", "night"
        ]

        for record in results:
            loc1 = record['location1'].lower()
            loc2 = record['location2'].lower()
            char_name = record['character_name'].lower()

            if any(x in loc1.split() for x in IGNORED_LOCATIONS) or any(x in loc2.split() for x in IGNORED_LOCATIONS):
                continue
            if char_name in loc1 or char_name in loc2:
                continue

            locs = sorted([record['location1'], record['location2']])
            key = f"{record['character_name']}_{locs[0]}_{locs[1]}"
            if key in seen_pairs: continue
            seen_pairs.add(key)

            premise = f"{record['character_name']} was at {record['location1']}."
            hypothesis = f"{record['character_name']} was at {record['location2']}."
            
            # print(f"   Analyzing Location: '{premise}' vs '{hypothesis}'...")
            nli_result = run_nli_check(premise, hypothesis)
            
            if nli_result["prediction"] == "contradiction" and nli_result["confidence"] > 0.5:
                inconsistencies.append(self._format_inconsistency(
                    "Temporal/Impossible Location",
                    {
                        "entity": record['character_name'],
                        "location_A": record['location1'],
                        "location_B": record['location2'],
                        "scope": record['conflict_time']
                    },
                    confidence=nli_result["confidence"]
                ))
        return inconsistencies

    # --- MODULE 2: INVENTORY (SEMANTIC UPGRADE) ---
    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies (Semantic Mode)...")
        
        # 1. Fetch ALL relationships involving the same person and item
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY, {})
        inconsistencies = []

        for record in results:
            # 2. Semantically Classify Predicates (No Hardcoded Lists)
            type_A = self._classify_predicate(record['character_name'], record['pred_A'], record['item_name'])
            type_B = self._classify_predicate(record['character_name'], record['pred_B'], record['item_name'])

            # 3. Check for Logic Conflict: Possession vs Loss
            # We only care if one implies Possession and the other implies Loss
            if (type_A == "POSSESSION" and type_B == "LOSS") or (type_A == "LOSS" and type_B == "POSSESSION"):
                
                # 4. Final Safety Check: Do they directly contradict?
                premise = f"{record['character_name']} {record['pred_A']} the {record['item_name']}."
                hypothesis = f"{record['character_name']} {record['pred_B']} the {record['item_name']}."
                
                print(f"   Conflict Candidate: ({type_A}) '{record['pred_A']}' vs ({type_B}) '{record['pred_B']}'")
                
                nli_result = run_nli_check(premise, hypothesis)
                
                # Accept contradiction OR high-confidence neutral if states are opposing
                if nli_result["prediction"] == "contradiction" or nli_result["confidence"] > 0.5:
                     inconsistencies.append(self._format_inconsistency(
                        "Inventory State Conflict",
                        {
                            "entity": record['character_name'],
                            "item": record['item_name'],
                            "state_A": f"{record['pred_A']} ({type_A})",
                            "state_B": f"{record['pred_B']} ({type_B})",
                            "scope": record['conflict_time']
                        },
                        confidence=nli_result["confidence"]
                    ))

        return inconsistencies

    def detect_location_inconsistency(self): return []
    def detect_identity_inconsistency(self): return []
    def detect_attribute_inconsistency(self): return []
    
    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        return all_conflicts