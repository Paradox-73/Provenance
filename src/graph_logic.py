from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check
from .friction_queries import IMPOSSIBLE_LOCATION_QUERY

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter

    def _format_inconsistency(self, conflict_type: str, details: dict) -> dict:
        """Helper to format inconsistency results."""
        return {
            "type": conflict_type,
            "description": f"Detected {conflict_type} inconsistency.",
            "details": details,
            "confidence": 0.0 # Will be updated with NLI or rule confidence
        }

    def detect_temporal_inconsistency(self) -> list[dict]:
        """
        Detects temporal inconsistencies, specifically an entity being in two different
        locations at the same "time" (as per IMPOSSIBLE_LOCATION_QUERY logic).
        """
        print("Detecting temporal/impossible location inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IMPOSSIBLE_LOCATION_QUERY)
        inconsistencies = []
        for record in results:
            # For a real system, you'd perform NLI on the conflicting statements
            # to confirm the contradiction semantically.
            premise = f"{record['character_name']} was at {record['location1']} at {record['conflict_time']} (source: {record['source1']})."
            hypothesis = f"{record['character_name']} was at {record['location2']} at {record['conflict_time']} (source: {record['source2']})."
            nli_result = run_nli_check(premise, hypothesis) if run_nli_check else {"prediction": "contradiction", "confidence": 0.9}

            if nli_result["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency(
                    "Temporal/Impossible Location",
                    {
                        "entity": record['character_name'],
                        "conflict_time": record['conflict_time'],
                        "location1": record['location1'],
                        "statement1_source": record['source1'],
                        "location2": record['location2'],
                        "statement2_source": record['source2'],
                        "nli_confidence": nli_result["confidence"]
                    }
                ))
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        """
        Detects inventory inconsistencies, e.g., an entity gains/loses an object
        without explanation across chapters.
        This is highly conceptual and would need very specific triples like "has", "loses", "gains".
        Simplified: Check if an entity 'has' an object in one chapter, but 'loses' it in another
        without a clear "event" explaining the loss.
        """
        print("Detecting inventory inconsistencies...")
        query = """
        MATCH (e:Entity)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(item1)
        WHERE a1.predicate = 'has'
        MATCH (e)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(item2)
        WHERE a2.predicate = 'loses' // Or an opposite state
        AND item1.name = item2.name
        AND a1.chapter_name <> a2.chapter_name // Across different chapters
        RETURN e.name AS entity, 
               item1.name AS item, 
               a1.provenance_text AS has_text, a1.chapter_name AS has_chapter,
               a2.provenance_text AS loses_text, a2.chapter_name AS loses_chapter
        """
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        for record in results:
            # Semantic check if 'loses' contradicts 'has' in absence of intervening event
            premise = f"{record['entity']} has {record['item']} in {record['has_chapter']}."
            hypothesis = f"{record['entity']} loses {record['item']} in {record['loses_chapter']}."
            nli_result = run_nli_check(premise, hypothesis) if run_nli_check else {"prediction": "contradiction", "confidence": 0.8}

            if nli_result["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency(
                    "Inventory",
                    {
                        "entity": record['entity'],
                        "item": record['item'],
                        "statement_has": record['has_text'],
                        "chapter_has": record['has_chapter'],
                        "statement_loses": record['loses_text'],
                        "chapter_loses": record['loses_chapter'],
                        "nli_confidence": nli_result["confidence"]
                    }
                ))
        return inconsistencies

    def detect_location_inconsistency(self) -> list[dict]:
        """
        Detects location inconsistencies, e.g., an immovable object changing location.
        This would require tagging 'immovable' objects.
        Simplified: Check if a "building" type entity is located at two different places.
        """
        print("Detecting location inconsistencies...")
        query = """
        MATCH (obj:Entity)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(loc1)
        WHERE a1.predicate = 'located at' AND obj.type = 'BUILDING' // Assuming 'BUILDING' is an immovable object type
        MATCH (obj)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(loc2)
        WHERE a2.predicate = 'located at'
        AND a1 <> a2
        AND loc1 <> loc2
        RETURN obj.name AS object, 
               loc1.name AS location1, a1.provenance_text AS text1, a1.chapter_name AS chapter1,
               loc2.name AS location2, a2.provenance_text AS text2, a2.chapter_name AS chapter2
        """
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        for record in results:
            premise = f"The {record['object']} is located at {record['location1']}."
            hypothesis = f"The {record['object']} is located at {record['location2']}."
            nli_result = run_nli_check(premise, hypothesis) if run_nli_check else {"prediction": "contradiction", "confidence": 0.95}

            if nli_result["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency(
                    "Location",
                    {
                        "object": record['object'],
                        "location1": record['location1'],
                        "statement1": record['text1'],
                        "chapter1": record['chapter1'],
                        "location2": record['location2'],
                        "statement2": record['text2'],
                        "chapter2": record['chapter2'],
                        "nli_confidence": nli_result["confidence"]
                    }
                ))
        return inconsistencies

    def detect_identity_inconsistency(self) -> list[dict]:
        """
        Detects identity inconsistencies, e.g., a person being referred to by a wrong name
        after coreference resolution. This is mainly caught by xCoRe, but could be a semantic
        contradiction if two canonical entities are asserted to be the same but have
        conflicting core attributes.
        Simplified: If two entities with different canonical_ids are both asserted to be "John Doe".
        """
        print("Detecting identity inconsistencies...")
        # This detection is more about cross-checking canonical_ids for same asserted name
        # or properties. This requires a more complex schema where 'name' is an attribute.
        # For now, let's assume if an entity is 'called' by two canonical names.
        query = """
        MATCH (e1:Entity)-[:HAS_ASSERTION]->(a1:Assertion)
        WHERE a1.predicate = 'is called'
        MATCH (e2:Entity)-[:HAS_ASSERTION]->(a2:Assertion)
        WHERE a2.predicate = 'is called'
        AND e1.canonical_id = e2.canonical_id // Same underlying entity
        AND a1.object <> a2.object // But different asserted names
        RETURN e1.name AS entity_name,
               a1.object.name AS name1, a1.provenance_text AS text1, a1.chapter_name AS chapter1,
               a2.object.name AS name2, a2.provenance_text AS text2, a2.chapter_name AS chapter2
        """
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        for record in results:
            premise = f"{record['entity_name']} is called {record['name1']}."
            hypothesis = f"{record['entity_name']} is called {record['name2']}."
            nli_result = run_nli_check(premise, hypothesis) if run_nli_check else {"prediction": "contradiction", "confidence": 0.88}

            if nli_result["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency(
                    "Identity",
                    {
                        "entity": record['entity_name'],
                        "asserted_name1": record['name1'],
                        "statement1": record['text1'],
                        "chapter1": record['chapter1'],
                        "asserted_name2": record['name2'],
                        "statement2": record['text2'],
                        "chapter2": record['chapter2'],
                        "nli_confidence": nli_result["confidence"]
                    }
                ))
        return inconsistencies


    def detect_attribute_inconsistency(self) -> list[dict]:
        """
        Detects attribute inconsistencies, e.g., a physical characteristic changing
        (eye color, height) without explanation.
        Simplified: An entity 'has attribute' X in one place and 'has attribute' Y for the same
        characteristic (e.g., 'eye color') in another.
        """
        print("Detecting attribute inconsistencies...")
        # This requires more explicit modeling of attributes as nodes or structured properties
        # For this example, we assume predicate itself carries the attribute info
        query = """
        MATCH (e:Entity)-[:HAS_ASSERTION]->(a1:Assertion)
        WHERE a1.predicate STARTS WITH 'has eye color'
        MATCH (e)-[:HAS_ASSERTION]->(a2:Assertion)
        WHERE a2.predicate STARTS WITH 'has eye color'
        AND a1 <> a2
        AND a1.predicate <> a2.predicate // Different eye colors asserted
        RETURN e.name AS entity,
               a1.predicate AS attribute1, a1.provenance_text AS text1, a1.chapter_name AS chapter1,
               a2.predicate AS attribute2, a2.provenance_text AS text2, a2.chapter_name AS chapter2
        """
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        for record in results:
            premise = f"{record['entity']} {record['attribute1']}."
            hypothesis = f"{record['entity']} {record['attribute2']}."
            nli_result = run_nli_check(premise, hypothesis) if run_nli_check else {"prediction": "contradiction", "confidence": 0.92}

            if nli_result["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency(
                    "Attribute",
                    {
                        "entity": record['entity'],
                        "attribute1": record['attribute1'],
                        "statement1": record['text1'],
                        "chapter1": record['chapter1'],
                        "attribute2": record['attribute2'],
                        "statement2": record['text2'],
                        "chapter2": record['chapter2'],
                        "nli_confidence": nli_result["confidence"]
                    }
                ))
        return inconsistencies
    
    def detect_all_inconsistencies(self) -> list[dict]:
        """Runs all inconsistency detection methods and returns a combined list."""
        all_conflicts = []
        all_conflicts.extend(self.detect_temporal_inconsistency())
        all_conflicts.extend(self.detect_inventory_inconsistency())
        all_conflicts.extend(self.detect_location_inconsistency())
        all_conflicts.extend(self.detect_identity_inconsistency())
        all_conflicts.extend(self.detect_attribute_inconsistency())
        return all_conflicts

if __name__ == '__main__':
    print("--- Conflict Detector Module Test ---")
    # This test assumes Neo4j is running and has some data ingested by neo4j_adapter test.
    # It also assumes nlp_pipeline's load_nli_model has been called if NLI checks are active.
    
    # Ensure a .env file exists with NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    # For testing, you might need to run neo4j_adapter.py's main block first
    # to populate the graph with some data.
    
    # Initialize Neo4jAdapter (assumes .env is set up)
    neo_adapter = Neo4jAdapter()
    try:
        neo_adapter.connect()
        # Optionally, load NLI model for semantic checks in detection functions
        # from src.nlp_pipeline import load_nli_model
        # load_nli_model()

        detector = ConflictDetector(neo_adapter)
        
        # Example: Inject a simple conflicting assertion for testing purposes
        # You'd normally ingest this via the NLP pipeline
        print("--- Injecting sample conflicting data for testing ---")
        with neo_adapter.driver.session() as session:
            session.run("MERGE (e:Entity {canonical_id: 'CAN_ENTITY_TEST'}) ON CREATE SET e.name='Test Entity'")
            session.run("""
                MERGE (e:Entity {canonical_id: 'CAN_ENTITY_TEST'})
                MERGE (locA:Entity {name: 'Location A'}) ON CREATE SET locA.type = 'LOCATION'
                MERGE (locB:Entity {name: 'Location B'}) ON CREATE SET locB.type = 'LOCATION'
                MERGE (a1:Assertion {id: 'test_triple_1', predicate: 'located at', chapter_name: 'Chapter X'})
                MERGE (a2:Assertion {id: 'test_triple_2', predicate: 'located at', chapter_name: 'Chapter X'})
                MERGE (e)-[:HAS_ASSERTION]->(a1)-[:ABOUT]->(locA)
                MERGE (e)-[:HAS_ASSERTION]->(a2)-[:ABOUT]->(locB)
            """)
            session.run("""
                MERGE (e:Entity {canonical_id: 'CAN_ENTITY_BUILDING'}) ON CREATE SET e.name='Grand Library', e.type='BUILDING'
                MERGE (locC:Entity {name: 'Main Street'}) ON CREATE SET locC.type = 'LOCATION'
                MERGE (locD:Entity {name: 'Oak Avenue'}) ON CREATE SET locD.type = 'LOCATION'
                MERGE (a3:Assertion {id: 'test_triple_3', predicate: 'located at', chapter_name: 'Chapter Y'})
                MERGE (a4:Assertion {id: 'test_triple_4', predicate: 'located at', chapter_name: 'Chapter Y'})
                MERGE (e)-[:HAS_ASSERTION]->(a3)-[:ABOUT]->(locC)
                MERGE (e)-[:HAS_ASSERTION]->(a4)-[:ABOUT]->(locD)
            """)
            session.run("""
                MERGE (e:Entity {canonical_id: 'CAN_ENTITY_PERSON_ATTR'}) ON CREATE SET e.name='Alice'
                MERGE (a5:Assertion {id: 'test_triple_5', predicate: 'has eye color blue', chapter_name: 'Chapter A'})
                MERGE (a6:Assertion {id: 'test_triple_6', predicate: 'has eye color brown', chapter_name: 'Chapter A'})
                MERGE (e)-[:HAS_ASSERTION]->(a5)
                MERGE (e)-[:HAS_ASSERTION]->(a6)
            """)
        print("Sample conflicting data injected.")


        print("--- Running all conflict detection ---")
        conflicts = detector.detect_all_inconsistencies()

        if conflicts:
            print("--- Detected Inconsistencies ---")
            for conflict in conflicts:
                print(f"Type: {conflict['type']}")
                print(f"Description: {conflict['description']}")
                for k, v in conflict['details'].items():
                    print(f"  {k}: {v}")
                print("-" * 20)
        else:
            print("No inconsistencies detected.")

    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        neo_adapter.close()
