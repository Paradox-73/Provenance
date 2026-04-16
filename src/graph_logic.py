from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check
from sentence_transformers import SentenceTransformer, util

# 1. TEMPORAL QUERY
IMPOSSIBLE_LOCATION_QUERY = """
MATCH (p:Entity)-[r1:LOCATED_AT]->(l1)
MATCH (p)-[r2:LOCATED_AT]->(l2)
WHERE id(l1) < id(l2) 
AND r1.chapter = r2.chapter 
AND r1.timestamp = r2.timestamp 
AND r1.timestamp <> 'unknown'
AND NOT (l1)-[:PART_OF*1..2]-(l2)
RETURN p.name as character_name, 
       l1.name as location1, 
       l2.name as location2, 
       r1.chapter as conflict_time, 
       r1.text as source1, 
       r2.text as source2
"""

# 2. INVENTORY QUERY - Updated to use plabel
INVENTORY_CONFLICT_QUERY = """
MATCH (p:Entity)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(item:Entity)
MATCH (p)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(item)
WHERE a1.chapter = a2.chapter
AND id(a1) < id(a2)
AND ((a1.label = 'POSSESSES' AND a2.label = 'LOST') OR (a1.label = 'LOST' AND a2.label = 'POSSESSES'))
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
MATCH (p:Entity)-[r1:HAS_IDENTITY]->(id1:Entity)
MATCH (p)-[r2:HAS_IDENTITY]->(id2:Entity)
WHERE id(id1) < id(id2)
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
MATCH (p:Entity)-[r1:HAS_ATTRIBUTE]->(attr1:Entity)
MATCH (p)-[r2:HAS_ATTRIBUTE]->(attr2:Entity)
WHERE id(attr1) < id(attr2)
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
        # Initialize semantic gatekeeper model
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

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
        if not results: return []
        
        inconsistencies = []
        seen_pairs = set()
        
        for record in results:
            try:
                char_name = record.get('character_name', 'The character')
                loc1 = str(record.get('location1', '')).lower()
                loc2 = str(record.get('location2', '')).lower()
                
                if not loc1 or not loc2: continue

                key = f"{char_name}_{sorted([loc1, loc2])[0]}_{sorted([loc1, loc2])[1]}"
                if key in seen_pairs: continue
                seen_pairs.add(key)

                nli_result = run_nli_check(f"{char_name} was at {record.get('location1')}.", f"{char_name} was at {record.get('location2')}.")
                if nli_result and nli_result.get("prediction") == "contradiction" and nli_result.get("confidence", 0) > 0.5:
                    inconsistencies.append(self._format_inconsistency("Temporal Location Conflict", record, nli_result["confidence"]))
            except Exception as e:
                print(f"[!] Error processing temporal conflict record: {e}")
                continue
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY)
        if not results: return []
        
        inconsistencies = []
        for record in results:
            # Direct graph traversal with label check eliminates need for NLI classification
            inconsistencies.append(self._format_inconsistency("Inventory Conflict", record, 1.0))
        return inconsistencies

    def detect_identity_inconsistency(self) -> list[dict]:
        print("Detecting identity inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IDENTITY_CONFLICT_QUERY)
        if not results: return []
        
        inconsistencies = []
        for record in results:
            try:
                char_name = record.get('character_name', 'The character')
                id1 = record.get('identity1', 'Unknown')
                id2 = record.get('identity2', 'Unknown')
                
                # OPTIMIZED: Smarter prompt for identity
                nli_result = run_nli_check(
                    f"{char_name} is officially identified as {id1}.", 
                    f"{char_name} is officially identified as {id2}."
                )
                if nli_result and nli_result.get("prediction") == "contradiction" and nli_result.get("confidence", 0) > 0.4:
                    inconsistencies.append(self._format_inconsistency("Identity Conflict", record, nli_result["confidence"]))
            except Exception as e:
                print(f"[!] Error processing identity conflict record: {e}")
                continue
        return inconsistencies

    def detect_attribute_inconsistency(self) -> list[dict]:
        print("Detecting attribute inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_CONFLICT_QUERY)
        if not results:
            print("DEBUG: No potential attribute conflicts found in graph.")
            return []
            
        print(f"DEBUG: Found {len(results)} potential attribute conflicts.")
        inconsistencies = []
        
        for record in results:
            try:
                print(f"DEBUG: Processing record: {record}")
                # Semantic Gatekeeper: Calculate similarity between attributes
                attr1 = record.get('attribute1')
                attr2 = record.get('attribute2')
                
                if attr1 is None or attr2 is None:
                    print(f"DEBUG: Skipping record with None attribute: attr1={attr1}, attr2={attr2}")
                    continue

                emb1 = self.similarity_model.encode(str(attr1), convert_to_tensor=True)
                emb2 = self.similarity_model.encode(str(attr2), convert_to_tensor=True)
                
                if emb1 is None or emb2 is None:
                    print(f"DEBUG: Skipping record due to encoding failure.")
                    continue
                    
                similarity = util.cos_sim(emb1, emb2).item()
                print(f"DEBUG: Similarity between '{attr1}' and '{attr2}': {similarity:.4f}")

                # ONLY execute NLI check if attributes belong to the same category (> 0.2 similarity)
                if similarity > 0.2:
                    char_name = record.get('character_name', 'The character')
                    nli_result = run_nli_check(
                        f"Physically, {char_name}'s appearance is described as having {attr1}.",
                        f"Physically, {char_name}'s appearance is described as having {attr2}."
                    )
                    print(f"DEBUG: NLI Result: {nli_result}")
                    if nli_result and nli_result.get("prediction") == "contradiction" and nli_result.get("confidence", 0) > 0.7:
                        inconsistencies.append(self._format_inconsistency("Attribute Conflict", record, nli_result["confidence"]))
            except Exception as e:
                print(f"[!] Error processing attribute conflict record: {e}")
                continue
                
        return inconsistencies
    
    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        all_conflicts.extend(self.detect_identity_inconsistency())
        all_conflicts.extend(self.detect_attribute_inconsistency())
        return all_conflicts