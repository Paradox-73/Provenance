from neo4j import GraphDatabase, basic_auth
import os
from dotenv import load_dotenv
import re

load_dotenv()

class Neo4jAdapter:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = None

    def connect(self):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=basic_auth(self.user, self.password))
            self.driver.verify_connectivity()
            print("Connected to Neo4j database successfully.")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            print("Neo4j connection closed.")

    def define_schema(self):
        if not self.driver: self.connect()
        print("Defining Neo4j schema...")
        with self.driver.session() as session:
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_id IS UNIQUE")
            session.run("CREATE INDEX IF NOT EXISTS FOR (t:Triple) ON (t.predicate)")

    def clear_database(self):
        if not self.driver: self.connect()
        print("Clearing Neo4j database...")
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def ingest_processed_data(self, processed_chapters: dict):
        if not self.driver: self.connect()
        print("Ingesting data into Neo4j...")
        
        with self.driver.session() as session:
            # --- PASS 1: Create Nodes ---
            print("Pass 1: Creating Nodes...")
            for chapter_name, chunks in processed_chapters.items():
                session.run("MERGE (c:Chapter {name: $name})", name=chapter_name)
                for chunk in chunks:
                    for entity in chunk["entities"]:
                        if "canonical_id" not in entity: continue
                        label = "Person" if entity["label"] == "PERSON" else "Location" if entity["label"] in ["LOC", "GPE", "FAC"] else "Entity"
                        session.run(f"MERGE (e:{label} {{canonical_id: $cid}}) ON CREATE SET e.name = $name", cid=entity["canonical_id"], name=entity["text"])
                    
                    for triple in chunk["triples"]:
                        for role in ["subject", "object"]:
                            item = triple[role]
                            if "canonical_id" not in item: item["canonical_id"] = f"UNRESOLVED_{hash(item['text'])}"
                            session.run("MERGE (e:Entity {canonical_id: $cid}) ON CREATE SET e.name = $name", cid=item["canonical_id"], name=item["text"])

            # --- PASS 2: Create Relationships ---
            print("Pass 2: Creating Relationships...")
            for chapter_name, chunks in processed_chapters.items():
                for chunk in chunks:
                    chunk_id = chunk["chunk_id"]
                    for triple in chunk["triples"]:
                        subj_id = triple["subject"]["canonical_id"]
                        obj_id = triple["object"]["canonical_id"]
                        pred_text = triple["predicate"]["text"].lower()
                        
                        # LOCATION LOGIC: Now saves 'chapter' explicitly
                        location_pattern = r"\b(at|in|on|puts you at|located)\b"
                        
                        if re.search(location_pattern, pred_text):
                            query = (
                                "MATCH (s {canonical_id: $sid}), (o {canonical_id: $oid}) "
                                "MERGE (s)-[r:LOCATED_AT]->(o) "
                                "ON CREATE SET r.timestamp = $ts, r.chapter = $chap, r.text = $txt"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, ts=f"{chapter_name}-{chunk_id}", chap=chapter_name, txt=triple["provenance"]["source_text_snippet"])
                        
                        # OTHER ASSERTIONS
                        else:
                            triple_id = f"trip_{hash(subj_id+pred_text+obj_id)}"
                            query = (
                                "MATCH (s {canonical_id: $sid}), (o {canonical_id: $oid}) "
                                "MERGE (s)-[:HAS_ASSERTION]->(a:Assertion {id: $tid}) "
                                "MERGE (a)-[:ABOUT]->(o) "
                                "SET a.predicate = $pred"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, tid=triple_id, pred=pred_text)
            print("Data ingestion complete.")

    def run_cypher_query(self, query: str, parameters: dict = None):
        if not self.driver: self.connect()
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]