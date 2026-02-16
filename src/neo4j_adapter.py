from neo4j import GraphDatabase, basic_auth
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

class Neo4jAdapter:
    def __init__(self, uri=None, user=None, password=None):
        """
        Initializes the Neo4jAdapter with connection details.
        Loads credentials from .env if not provided.
        """
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password") # Change default password!
        self.driver = None

    def connect(self):
        """Establishes a connection to the Neo4j database."""
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            self.driver.verify_connectivity()
            print("Connected to Neo4j database successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        """Closes the Neo4j database connection."""
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def define_schema(self):
        """
        Defines the Neo4j graph schema including constraints and indexes
        for efficient querying and data integrity.
        """
        if not self.driver:
            self.connect()

        print("Defining Neo4j schema (constraints and indexes)...")
        schema_queries = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.canonical_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (l:Location) REQUIRE l.canonical_id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chapter) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (t:Triple) REQUIRE t.id IS UNIQUE", # Unique ID for each triple
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Triple) ON (t.predicate)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Triple) ON (t.chapter_name)",
            "CREATE INDEX IF NOT EXISTS FOR (t:Triple) ON (t.provenance_confidence)"
        ]

        with self.driver.session() as session:
            for query in schema_queries:
                session.run(query)
        print("Neo4j schema definition complete.")

    def clear_database(self):
        """
        Deletes all nodes and relationships from the Neo4j database.
        """
        if not self.driver:
            self.connect()

        print("Clearing the Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Neo4j database cleared.")

    def ingest_processed_data(self, processed_chapters: dict):
        """
        Ingests processed NLP data into the Neo4j graph using a robust two-pass approach.
        1. First pass: Create all nodes (Chapters, Entities).
        2. Second pass: Create all relationships between the nodes.
        """
        if not self.driver:
            self.connect()

        print("Ingesting processed NLP data into Neo4j...")
        with self.driver.session() as session:
            # --- PASS 1: CREATE ALL NODES ---
            print("Pass 1: Creating all Chapter and Entity nodes...")
            for chapter_name, chunks in processed_chapters.items():
                # Ensure Chapter node exists
                session.run("MERGE (c:Chapter {name: $chapter_name})", chapter_name=chapter_name)
                
                for chunk in chunks:
                    for entity in chunk["entities"]:
                        if "canonical_id" not in entity or "UNRESOLVED" in entity["canonical_id"]:
                            continue # Skip creating nodes for unresolved entities

                        node_label = "Entity"
                        if entity["label"] == "PERSON":
                            node_label = "Person"
                        elif entity["label"] in ["LOC", "GPE"]:
                            node_label = "Location"
                        elif entity["label"] == "ORG":
                            node_label = "Organization"
                        elif entity["label"] == "FAC":
                            node_label = "Facility"

                        session.run(
                            f"MERGE (e:{node_label} {{canonical_id: $canonical_id}}) "
                            "ON CREATE SET e.name = $name, e.type = $type",
                            canonical_id=entity["canonical_id"],
                            name=entity["text"],
                            type=entity["label"]
                        )
            
            # --- PASS 2: CREATE ALL RELATIONSHIPS ---
            print("Pass 2: Creating all relationships between nodes...")
            for chapter_name, chunks in processed_chapters.items():
                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    
                    # Create MENTIONED_IN relationships
                    for entity in chunk["entities"]:
                        if "canonical_id" in entity and "UNRESOLVED" not in entity["canonical_id"]:
                            session.run(
                                "MATCH (e {canonical_id: $canonical_id}), (c:Chapter {name: $chapter_name}) "
                                "MERGE (e)-[:MENTIONED_IN]->(c)",
                                canonical_id=entity["canonical_id"],
                                chapter_name=chapter_name
                            )

                    # Create relationships for triples
                    for triple in chunk["triples"]:
                        subj_canonical_id = triple["subject"].get("canonical_id")
                        obj_canonical_id = triple["object"].get("canonical_id")

                        # VALIDATION: Ensure both subject and object are resolved before creating a relationship
                        if not subj_canonical_id or "UNRESOLVED" in subj_canonical_id:
                            print(f"WARNING: Skipping triple due to unresolved subject: {triple['subject']['text']}")
                            continue
                        if not obj_canonical_id or "UNRESOLVED" in obj_canonical_id:
                            print(f"WARNING: Skipping triple due to unresolved object: {triple['object']['text']}")
                            continue
                        
                        # At this point, both subject and object are resolved entities.
                        predicate_text = triple["predicate"]["text"].lower()
                        
                        # --- Heuristic for LOCATED_AT relationship ---
                        location_predicates = ["lives in", "stood on", "made her way towards", "found herself in", "ran through", "frequented", "was with", "in"]
                        # We need to know if the OBJECT is a location. We can't know the label directly here,
                        # so we make the query match against the Location label.
                        is_location_triple = any(p in predicate_text for p in location_predicates)

                        if is_location_triple:
                            timestamp = f"{chapter_name}-{chunk_id}"
                            # The MATCH here is more specific, looking for a Location node for the object.
                            result = session.run(
                                "MATCH (s {canonical_id: $subj_id}), (o:Location {canonical_id: $obj_id}) "
                                "MERGE (s)-[r:LOCATED_AT]->(o) "
                                "ON CREATE SET r.timestamp = $timestamp, r.source_text = $source, r.confidence = $conf, r.chapter = $chap",
                                subj_id=subj_canonical_id,
                                obj_id=obj_canonical_id,
                                timestamp=timestamp,
                                source=triple["provenance"]["source_text_snippet"],
                                conf=triple["provenance"]["confidence"],
                                chap=chapter_name
                            )
                            if result.summary().counters.relationships_created == 0:
                                # This can happen if the object node wasn't a Location.
                                print(f"INFO: LOCATED_AT for '{predicate_text}' not created, object node may not be a Location.")
                        
                        # --- Generic Assertion node for all other triples ---
                        else:
                            subj_id = triple["subject"].get("canonical_id")
                            obj_id = triple["object"].get("canonical_id")
                            pred_text = triple["predicate"]["text"]
                            provenance_snippet = triple["provenance"]["source_text_snippet"]
                            unique_hashable_tuple = (subj_id, pred_text, obj_id, provenance_snippet)
                            triple_id = f"{chapter_name}_chunk{chunk_id}_triple_{hash(unique_hashable_tuple)}"

                            session.run(
                                "MATCH (s {canonical_id: $subj_id}), (o {canonical_id: $obj_id}) "
                                "MERGE (a:Assertion {id: $triple_id}) "
                                "ON CREATE SET a.predicate = $predicate, a.provenance_text = $provenance, a.confidence = $conf, a.chapter = $chap "
                                "MERGE (s)-[:HAS_ASSERTION]->(a) "
                                "MERGE (a)-[:ABOUT]->(o)",
                                subj_id=subj_canonical_id,
                                obj_id=obj_canonical_id,
                                triple_id=triple_id,
                                predicate=pred_text,
                                provenance=provenance_snippet,
                                conf=triple["provenance"]["confidence"],
                                chap=chapter_name
                            )

            print("Data ingestion into Neo4j complete.")

    def run_cypher_query(self, query: str, parameters: dict = None):
        """
        Runs a generic Cypher query and returns results.
        """
        if not self.driver:
            self.connect()
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

if __name__ == '__main__':
    print("--- Neo4j Adapter Module Test ---")
    # This is a simplified test and may not reflect the full complexity of the pipeline.
    # It's recommended to run the main pipeline for a full integration test.
    adapter = Neo4jAdapter()
    try:
        adapter.connect()
        adapter.clear_database()
        adapter.define_schema()
        # ... simplified test data could be added here ...
        print("Test concluded. Run the main application for a full test.")
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        adapter.close()
