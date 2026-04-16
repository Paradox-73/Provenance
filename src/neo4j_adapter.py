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

    def _normalize_timestamp(self, ts: str) -> str:
        if not ts: return "unknown"
        match = re.search(r'(\d{2}:\d{2})', ts)
        return match.group(1) if match else ts

    def ingest_processed_data(self, processed_chapters: dict):
        if not self.driver: self.connect()
        print("Ingesting data into Neo4j...")
        
        with self.driver.session() as session:
            # --- PASS 1: Create Nodes ---
            print("Pass 1: Creating Nodes...")
            for chapter_name, chunks in processed_chapters.items():
                session.run("MERGE (c:Chapter {name: $name})", name=chapter_name)
                for chunk in chunks:
                    for entity in chunk.get("entities", []):
                        if "canonical_id" not in entity: continue
                        label = "Person" if entity.get("label") == "PERSON" else "Location" if entity.get("label") in ["LOC", "GPE", "FAC"] else "Entity"
                        session.run(f"MERGE (e:Entity {{canonical_id: $cid}}) ON CREATE SET e.name = $name SET e:{label}", cid=entity["canonical_id"], name=entity.get("text", "Unknown"))
                    
                    for triple in chunk.get("triples", []):
                        for role in ["subject", "object"]:
                            item = triple.get(role, {})
                            if not item: continue
                            if "canonical_id" not in item: 
                                item["canonical_id"] = f"UNRESOLVED_{hash(item.get('text', 'unknown'))}"
                            session.run("MERGE (e:Entity {canonical_id: $cid}) ON CREATE SET e.name = $name", cid=item["canonical_id"], name=item.get("text", "Unknown"))

            # --- PASS 2: Create Relationships ---
            print("Pass 2: Creating Relationships...")
            for chapter_name, chunks in processed_chapters.items():
                for chunk in chunks:
                    for triple in chunk.get("triples", []):
                        subj_id = triple.get("subject", {}).get("canonical_id", "UNKNOWN")
                        obj_id = triple.get("object", {}).get("canonical_id", "UNKNOWN")
                        pred_label = triple.get("predicate", {}).get("label", "UNKNOWN")
                        pred_text = triple.get("predicate", {}).get("text", "").lower()
                        source_text = triple.get("provenance", {}).get("source_text_snippet", "Source not provided")
                        narrative_time = self._normalize_timestamp(triple.get("timestamp", "unknown"))

                        if source_text == "Source not provided" or not pred_text:
                            continue

                        # Handle Directionality for POSSESSES/LOST
                        if pred_label in ["POSSESSES", "LOST", "NOT_POSSESSES", "NOT_LOST"]:
                            if any(name in obj_id for name in ["KAEL", "MARIA", "ARIS", "EVANS", "CAPTAIN"]):
                                subj_id, obj_id = obj_id, subj_id

                        # Map labels to direct relationships (including NOT_ variants)
                        if pred_label in ["LOCATED_AT", "NOT_LOCATED_AT"]:
                            query = (
                                f"MERGE (s:Entity {{canonical_id: $sid}}) "
                                f"MERGE (o:Entity {{canonical_id: $oid}}) "
                                f"MERGE (s)-[r:{pred_label}]->(o) "
                                "SET r.timestamp = $ts, r.chapter = $chap, r.text = $txt"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, ts=narrative_time, chap=chapter_name, txt=source_text)
                        
                        elif pred_label in ["PART_OF", "NOT_PART_OF"]:
                            query = (
                                f"MERGE (s:Entity {{canonical_id: $sid}}) "
                                f"MERGE (o:Entity {{canonical_id: $oid}}) "
                                f"MERGE (s)-[r:{pred_label}]->(o) "
                                "SET r.chapter = $chap, r.text = $txt"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, chap=chapter_name, txt=source_text)

                        elif pred_label in ["IDENTITY", "HAS_IDENTITY", "NOT_HAS_IDENTITY"]:
                            # Normalize label to HAS_IDENTITY or NOT_HAS_IDENTITY
                            rel_type = "HAS_IDENTITY" if "NOT" not in pred_label else "NOT_HAS_IDENTITY"
                            query = (
                                f"MERGE (s:Entity {{canonical_id: $sid}}) "
                                f"MERGE (o:Entity {{canonical_id: $oid}}) "
                                f"MERGE (s)-[r:{rel_type}]->(o) "
                                "SET r.chapter = $chap, r.text = $txt"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, chap=chapter_name, txt=source_text)

                        elif pred_label in ["ATTRIBUTE", "HAS_ATTRIBUTE", "NOT_HAS_ATTRIBUTE"]:
                            rel_type = "HAS_ATTRIBUTE" if "NOT" not in pred_label else "NOT_HAS_ATTRIBUTE"
                            query = (
                                f"MERGE (s:Entity {{canonical_id: $sid}}) "
                                f"MERGE (o:Entity {{canonical_id: $oid}}) "
                                f"MERGE (s)-[r:{rel_type}]->(o) "
                                "SET r.chapter = $chap, r.text = $txt, r.target_feature = $tf"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, chap=chapter_name, txt=source_text, tf=triple.get("target_feature", "general"))

                        else:
                            triple_id = f"trip_{hash(subj_id+pred_text+obj_id)}"
                            query = (
                                "MERGE (s:Entity {canonical_id: $sid}) "
                                "MERGE (o:Entity {canonical_id: $oid}) "
                                "MERGE (s)-[:HAS_ASSERTION]->(a:Assertion {id: $tid}) "
                                "MERGE (a)-[:ABOUT]->(o) "
                                "SET a.predicate = $pred, a.label = $plabel, a.chapter = $chap, a.text = $txt"
                            )
                            session.run(query, sid=subj_id, oid=obj_id, tid=triple_id, pred=pred_text, plabel=pred_label, chap=chapter_name, txt=source_text)
            print("Data ingestion complete.")

    def run_cypher_query(self, query: str, parameters: dict = None):
        if not self.driver: self.connect()
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                if result is None:
                    print(f"[!] Warning: Cypher query returned None: {query}")
                    return []
                return [record for record in result]
        except Exception as e:
            print(f"[!] Error running Cypher query: {e}")
            print(f"Query: {query}")
            return []
