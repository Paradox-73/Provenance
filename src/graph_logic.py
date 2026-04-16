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

# 2. INVENTORY QUERY
INVENTORY_CONFLICT_QUERY = """
MATCH (p:Entity)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(item:Entity)
MATCH (p)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(item)
WHERE a1.chapter = a2.chapter
AND id(a1) < id(a2)
AND a1.text <> a2.text
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
AND r1.text <> r2.text
RETURN p.name as character_name,
       id1.name as identity1,
       id2.name as identity2,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

# 4. ATTRIBUTE QUERY (Phase 3: Strict Attribute Graph Logic)
ATTRIBUTE_CONFLICT_QUERY = """
MATCH (p:Entity)-[r1:HAS_ATTRIBUTE]->(attr1:Entity)
MATCH (p)-[r2:HAS_ATTRIBUTE]->(attr2:Entity)
WHERE id(attr1) < id(attr2)
AND r1.chapter = r2.chapter
AND r1.target_feature = r2.target_feature
AND r1.text <> r2.text
RETURN p.name as character_name,
       attr1.name as attribute1,
       attr2.name as attribute2,
       r1.target_feature as target_feature,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

# 5. ATTRIBUTE TRANSFER QUERY
ATTRIBUTE_TRANSFER_QUERY = """
MATCH (p1:Entity)-[r1:HAS_ATTRIBUTE]->(attr:Entity)
MATCH (p2:Entity)-[r2:HAS_ATTRIBUTE]->(attr)
WHERE id(p1) < id(p2)
RETURN p1.name as char1,
       p2.name as char2,
       attr.name as attribute,
       r1.chapter as chap1,
       r2.chapter as chap2,
       r1.text as source1,
       r2.text as source2
"""

# 6. NEGATION CONFLICT QUERY
NEGATION_CONFLICT_QUERY = """
MATCH (p:Entity)-[r1:HAS_IDENTITY]->(target:Entity)
MATCH (p)-[r2:NOT_HAS_IDENTITY]->(target)
WHERE r1.chapter = r2.chapter
RETURN p.name as character_name,
       target.name as target_name,
       r1.chapter as conflict_time,
       r1.text as source_pos,
       r2.text as source_neg
"""

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter
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
        groups = {}
        for record in results:
            key = f"{record['character_name']}_{record['conflict_time']}"
            if key not in groups: groups[key] = {"char": record['character_name'], "time": record['conflict_time'], "locs": set(), "sources": set()}
            groups[key]["locs"].update([record['location1'], record['location2']])
            groups[key]["sources"].update([record['source1'], record['source2']])
        inconsistencies = []
        for key, group in groups.items():
            locs = list(group["locs"])
            if len(locs) < 2: continue
            nli = run_nli_check(f"{group['char']} was at {locs[0]}.", f"{group['char']} was at {locs[1]}.")
            if nli and nli.get("prediction") == "contradiction" and nli.get("confidence", 0) > 0.5:
                inconsistencies.append(self._format_inconsistency("Temporal Location Conflict", {"character": group['char'], "time": group['time'], "locations": locs, "sources": list(group['sources'])}, nli["confidence"]))
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY)
        if not results: return []
        groups = {}
        for record in results:
            key = f"{record['character_name']}_inventory_{record['item_name']}"
            if key not in groups: groups[key] = {"char": record['character_name'], "item": record['item_name'], "sources": set(), "chapter": record['conflict_time']}
            groups[key]["sources"].update([record['source1'], record['source2']])
        inconsistencies = []
        for key, group in groups.items():
            inconsistencies.append(self._format_inconsistency("Inventory Conflict", {"character": group['char'], "item": group['item'], "sources": list(group['sources']), "chapter": group['chapter']}, 1.0))
        return inconsistencies

    def detect_identity_inconsistency(self) -> list[dict]:
        print("Detecting identity inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IDENTITY_CONFLICT_QUERY)
        if not results: return []
        groups = {}
        for record in results:
            key = f"{record['character_name']}_identity"
            if key not in groups: groups[key] = {"char": record['character_name'], "identities": set(), "sources": set()}
            groups[key]["identities"].update([record['identity1'], record['identity2']])
            groups[key]["sources"].update([record['source1'], record['source2']])
        inconsistencies = []
        for key, group in groups.items():
            ids = list(group["identities"])
            if len(ids) < 2: continue
            nli = run_nli_check(f"{group['char']} is described as {ids[0]}.", f"{group['char']} is described as {ids[1]}.")
            if nli and nli.get("prediction") == "contradiction" and nli.get("confidence", 0) > 0.4:
                inconsistencies.append(self._format_inconsistency("Identity Conflict", {"character": group['char'], "identities": ids, "sources": list(group['sources'])}, nli["confidence"]))
        return inconsistencies

    def detect_attribute_inconsistency(self) -> list[dict]:
        print("Detecting attribute inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_CONFLICT_QUERY)
        if not results: return []
        groups = {}
        for record in results:
            key = f"{record['character_name']}_{record['target_feature']}"
            if key not in groups: groups[key] = {"char": record['character_name'], "feature": record['target_feature'], "pairs": []}
            groups[key]["pairs"].append(record)
        inconsistencies = []
        for char_key, group in groups.items():
            processed = set()
            for record in group["pairs"]:
                pair = tuple(sorted([record['attribute1'], record['attribute2']]))
                if pair in processed: continue
                processed.add(pair)
                emb1 = self.similarity_model.encode(str(record['attribute1']), convert_to_tensor=True)
                emb2 = self.similarity_model.encode(str(record['attribute2']), convert_to_tensor=True)
                
                # LOWERED THRESHOLD back to 0.2 as per Phase 3 instructions
                if util.cos_sim(emb1, emb2).item() > 0.2:
                    nli = run_nli_check(f"{record['character_name']} is described as {record['attribute1']}.", f"{record['character_name']} is described as {record['attribute2']}.")
                    if nli and nli.get("prediction") == "contradiction" and nli.get("confidence", 0) > 0.7:
                        inconsistencies.append(self._format_inconsistency("Attribute Conflict", {"character": record['character_name'], "feature": record['target_feature'], "attr1": record['attribute1'], "attr2": record['attribute2'], "sources": [record['source1'], record['source2']]}, nli["confidence"]))
        return inconsistencies

    def detect_attribute_transfer(self) -> list[dict]:
        print("Detecting impossible attribute transfers...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_TRANSFER_QUERY)
        if not results: return []
        inconsistencies = []
        for record in results:
            inconsistencies.append(self._format_inconsistency("Attribute Transfer Conflict", {"char1": record['char1'], "char2": record['char2'], "attribute": record['attribute'], "sources": [record['source1'], record['source2']]}, 1.0))
        return inconsistencies

    def detect_negation_conflict(self) -> list[dict]:
        print("Detecting negation-based contradictions...")
        results = self.neo4j_adapter.run_cypher_query(NEGATION_CONFLICT_QUERY)
        if not results: return []
        inconsistencies = []
        for record in results:
            inconsistencies.append(self._format_inconsistency("Negation Conflict", {"character": record['character_name'], "target": record['target_name'], "sources": [record['source_pos'], record['source_neg']]}, 1.0))
        return inconsistencies
    
    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        all_conflicts.extend(self.detect_identity_inconsistency())
        all_conflicts.extend(self.detect_attribute_inconsistency())
        all_conflicts.extend(self.detect_attribute_transfer())
        all_conflicts.extend(self.detect_negation_conflict())
        return all_conflicts
