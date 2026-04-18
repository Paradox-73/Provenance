from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check
from sentence_transformers import SentenceTransformer, util

# 1. TEMPORAL QUERY (Hierarchy-Aware)
IMPOSSIBLE_LOCATION_QUERY = """
MATCH (p:Entity)-[r1:LOCATED_AT]->(l1)
MATCH (p)-[r2:LOCATED_AT]->(l2)
WHERE id(l1) < id(l2) 
AND r1.chapter = r2.chapter 
AND (r1.timestamp = r2.timestamp OR (r1.timestamp <> 'unknown' AND r2.timestamp <> 'unknown'))
// Check if they are on different decks (Impossible Teleportation)
OPTIONAL MATCH (l1)-[:PART_OF]->(d1:Deck)
OPTIONAL MATCH (l2)-[:PART_OF]->(d2:Deck)
WITH p, l1, l2, r1, r2, d1, d2
WHERE (d1 IS NOT NULL AND d2 IS NOT NULL AND d1 <> d2) OR NOT (l1)-[:PART_OF*1..2]-(l2)
RETURN p.name as character_name, 
       l1.name as location1, 
       l2.name as location2, 
       r1.chapter as conflict_time, 
       r1.text as source1, 
       r2.text as source2,
       r1.timestamp as ts1,
       r2.timestamp as ts2
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

# 3. IDENTITY QUERY (Role-Aware)
IDENTITY_CONFLICT_QUERY = """
MATCH (p:Entity)-[r1:HAS_ROLE]->(role1:Role)
MATCH (p)-[r2:HAS_ROLE]->(role2:Role)
WHERE id(role1) < id(role2)
AND role1.group = role2.group
RETURN p.name as character_name,
       role1.name as identity1,
       role2.name as identity2,
       'Implicit Scene Logic' as source1,
       'Role Overlap' as source2,
       'Multiple' as conflict_time
UNION
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

# 4. ATTRIBUTE QUERY (Extended)
ATTRIBUTE_CONFLICT_QUERY = """
MATCH (p:Entity)-[r1:HAS_ATTRIBUTE]->(attr1:Entity)
MATCH (p)-[r2:HAS_ATTRIBUTE]->(attr2:Entity)
WHERE id(attr1) < id(attr2)
AND r1.chapter = r2.chapter
RETURN p.name as character_name,
       attr1.name as attribute1,
       attr2.name as attribute2,
       r1.target_feature as target_feature,
       r1.text as source1,
       r2.text as source2,
       r1.chapter as conflict_time
"""

# 5. ATTRIBUTE THEFT (The "Unique Attribute" Query)
ATTRIBUTE_THEFT_QUERY = """
MATCH (p1:Entity)-[r1:HAS_ATTRIBUTE]->(attr:Entity)
MATCH (p2:Entity)-[r2:HAS_ATTRIBUTE]->(attr)
WHERE p1.canonical_id <> p2.canonical_id
// Only flag "Unique" sounding attributes like scars or specific items
AND (attr.name CONTAINS 'scar' OR attr.name CONTAINS 'Key' OR attr.name CONTAINS 'Artifact')
RETURN p1.name as original_owner,
       p2.name as new_owner,
       attr.name as stolen_trait,
       r1.chapter as chap1,
       r2.chapter as chap2,
       r1.text as source1,
       r2.text as source2
"""

# 7. HISTORICAL CONTINUITY (Cross-Chapter State Drift)
HISTORICAL_DRIFT_QUERY = """
MATCH (e:Entity)-[r1]->(state1)
MATCH (e)-[r2]->(state2)
WHERE r1.chapter < r2.chapter
AND type(r1) = type(r2)
AND state1.name <> state2.name
AND type(r1) IN ['HAS_ROLE', 'HAS_IDENTITY']
RETURN e.name as entity_name,
       state1.name as early_state,
       state2.name as late_state,
       r1.chapter as chap_early,
       r2.chapter as chap_late,
       r1.text as source_early,
       r2.text as source_late
"""

# 8. ENVIRONMENTAL FLUX (Light/Vessel Mutations)
ENVIRONMENTAL_FLUX_QUERY = """
MATCH (env:Entity)-[r1:HAS_ATTRIBUTE]->(s1)
MATCH (env)-[r2:HAS_ATTRIBUTE]->(s2)
WHERE (env:Location OR env.canonical_id CONTAINS 'LIGHT' OR env.canonical_id CONTAINS 'SHIP' OR env.canonical_id CONTAINS 'VESSEL')
AND r1.chapter = r2.chapter
AND s1.name <> s2.name
RETURN env.name as environment_item,
       s1.name as state1,
       s2.name as state2,
       r1.chapter as chap,
       r1.text as source1,
       r2.text as source2
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
        inconsistencies = []
        for record in results:
            nli = run_nli_check(f"{record['character_name']} was at {record['location1']}.", f"{record['character_name']} was at {record['location2']}.")
            if nli and nli.get("prediction") == "contradiction":
                inconsistencies.append(self._format_inconsistency("Temporal Location Conflict", record, nli["confidence"]))
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY)
        return [self._format_inconsistency("Inventory Conflict", r, 1.0) for r in results]

    def detect_identity_inconsistency(self) -> list[dict]:
        print("Detecting identity/role inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IDENTITY_CONFLICT_QUERY)
        return [self._format_inconsistency("Identity Conflict", r, 1.0) for r in results]

    def detect_attribute_inconsistency(self) -> list[dict]:
        print("Detecting attribute inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_CONFLICT_QUERY)
        inconsistencies = []
        for record in results:
            nli = run_nli_check(f"{record['character_name']} is {record['attribute1']}.", f"{record['character_name']} is {record['attribute2']}.")
            if nli and nli.get("prediction") == "contradiction":
                inconsistencies.append(self._format_inconsistency("Attribute Conflict", record, nli["confidence"]))
        return inconsistencies

    def detect_attribute_theft(self) -> list[dict]:
        print("Detecting unique attribute theft...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_THEFT_QUERY)
        return [self._format_inconsistency("Attribute Transfer", r, 1.0) for r in results]

    def detect_historical_drift(self) -> list[dict]:
        print("Detecting cross-chapter state drift...")
        results = self.neo4j_adapter.run_cypher_query(HISTORICAL_DRIFT_QUERY)
        return [self._format_inconsistency("Historical Drift", r, 1.0) for r in results]

    def detect_environmental_flux(self) -> list[dict]:
        print("Detecting environmental mutations...")
        results = self.neo4j_adapter.run_cypher_query(ENVIRONMENTAL_FLUX_QUERY)
        return [self._format_inconsistency("Environmental Mutation", r, 1.0) for r in results]

    def detect_all_inconsistencies(self) -> list[dict]:
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        all_conflicts.extend(self.detect_identity_inconsistency())
        all_conflicts.extend(self.detect_attribute_inconsistency())
        all_conflicts.extend(self.detect_attribute_theft())
        all_conflicts.extend(self.detect_historical_drift())
        all_conflicts.extend(self.detect_environmental_flux())
        return all_conflicts
