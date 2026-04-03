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

# 2. INVENTORY QUERY
INVENTORY_CONFLICT_QUERY = """
MATCH (p:Person)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(item:Entity)
MATCH (p)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(item)
WHERE a1.chapter = a2.chapter
AND a1.id <> a2.id
RETURN p.name as character_name,
       item.name as item_name,
       a1.predicate as pred_A,
       a2.predicate as pred_B,
       a1.text as source1,
       a2.text as source2,
       a1.chapter as conflict_time
"""

# 3. IDENTITY QUERY
IDENTITY_CONFLICT_QUERY = """
MATCH (p:Person)-[r1:HAS_IDENTITY]->(id1:Entity)
MATCH (p)-[r2:HAS_IDENTITY]->(id2:Entity)
WHERE id1 <> id2
AND r1.chapter = r2.chapter
RETURN p.name as character_name,
       id1.name as identity1,
       id2.name as identity2,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

# 4. ATTRIBUTE QUERY
ATTRIBUTE_CONFLICT_QUERY = """
MATCH (p:Person)-[r1:HAS_ATTRIBUTE]->(attr1:Entity)
MATCH (p)-[r2:HAS_ATTRIBUTE]->(attr2:Entity)
WHERE attr1 <> attr2
AND r1.chapter = r2.chapter
RETURN p.name as character_name,
       attr1.name as attribute1,
       attr2.name as attribute2,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter
        self.verb_cache = {}

    def _format_inconsistency(self, conflict_type: str, details: dict, confidence: float) -> dict:
        return {
            "type": conflict_type,
            "description": f"Detected {conflict_type} inconsistency.",
            "details": details,
            "confidence": confidence
        }

    def _classify_predicate(self, entity, predicate, item) -> str:
        cache_key = predicate.lower()
        if cache_key in self.verb_cache: return self.verb_cache[cache_key]

        premise = f"{entity} {predicate} the {item}."
        pos_result = run_nli_check(premise, f"{entity} possesses the {item}.")
        loss_result = run_nli_check(premise, f"{entity} does not possess the {item}.")

        classification = "OTHER"
        if pos_result["prediction"] == "entailment" and pos_result["confidence"] > 0.5:
            classification = "POSSESSION"
        elif loss_result["prediction"] == "entailment" and loss_result["confidence"] > 0.5:
            classification = "LOSS"
        
        if any(v in predicate for v in ["lost", "dropped", "missing"]): classification = "LOSS"
        self.verb_cache[cache_key] = classification
        return classification

    def detect_temporal_inconsistency(self) -> list[dict]:
        print("Detecting temporal/impossible location inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IMPOSSIBLE_LOCATION_QUERY)
        inconsistencies = []
        seen_pairs = set()
        
        for record in results:
            loc1, loc2 = record['location1'].lower(), record['location2'].lower()
            
            key = f"{record['character_name']}_{sorted([loc1, loc2])[0]}_{sorted([loc1, loc2])[1]}"
            if key in seen_pairs: continue
            seen_pairs.add(key)

            nli_result = run_nli_check(f"{record['character_name']} was at {record['location1']}.", f"{record['character_name']} was at {record['location2']}.")
            if nli_result["prediction"] == "contradiction" and nli_result["confidence"] > 0.5:
                inconsistencies.append(self._format_inconsistency("Temporal Location Conflict", record, nli_result["confidence"]))
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY)
        inconsistencies = []
        for record in results:
            type_A = self._classify_predicate(record['character_name'], record['pred_A'], record['item_name'])
            type_B = self._classify_predicate(record['character_name'], record['pred_B'], record['item_name'])
            if (type_A == "POSSESSION" and type_B == "LOSS") or (type_A == "LOSS" and type_B == "POSSESSION"):
                nli_result = run_nli_check(f"{record['character_name']} {record['pred_A']} the {record['item_name']}.", f"{record['character_name']} {record['pred_B']} the {record['item_name']}.")
                if nli_result["prediction"] == "contradiction" or nli_result["confidence"] > 0.5:
                    inconsistencies.append(self._format_inconsistency("Inventory Conflict", record, nli_result["confidence"]))
        return inconsistencies

    def detect_identity_inconsistency(self) -> list[dict]:
        print("Detecting identity inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IDENTITY_CONFLICT_QUERY)
        inconsistencies = []
        for record in results:
            nli_result = run_nli_check(f"{record['character_name']} is {record['identity1']}.", f"{record['character_name']} is {record['identity2']}.")
            if nli_result["prediction"] == "contradiction" and nli_result["confidence"] > 0.4:
                inconsistencies.append(self._format_inconsistency("Identity Conflict", record, nli_result["confidence"]))
        return inconsistencies

    def detect_attribute_inconsistency(self) -> list[dict]:
        print("Detecting attribute inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_CONFLICT_QUERY)
        inconsistencies = []
        for record in results:
            nli_result = run_nli_check(f"{record['character_name']} has {record['attribute1']}.", f"{record['character_name']} has {record['attribute2']}.")
            if nli_result["prediction"] == "contradiction" and nli_result["confidence"] > 0.4:
                inconsistencies.append(self._format_inconsistency("Attribute Conflict", record, nli_result["confidence"]))
        return inconsistencies
    
    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        all_conflicts.extend(self.detect_identity_inconsistency())
        all_conflicts.extend(self.detect_attribute_inconsistency())
        return all_conflicts