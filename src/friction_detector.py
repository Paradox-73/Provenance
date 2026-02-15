import json
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

# Assuming friction_queries.py is in the same directory
from friction_queries import IMPOSSIBLE_LOCATION_QUERY

class FrictionDetector:
    """
    A class to detect logical friction in a Neo4j graph.
    Connects to a Neo4j database, loads data, and runs Cypher queries.
    """

    def __init__(self, uri, user, password):
        """
        Initializes the detector and connects to the Neo4j database.

        Args:
            uri (str): The Bolt URI for the Neo4j instance.
            user (str): The username for the database.
            password (str): The password for the database.
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        """Closes the database connection."""
        self.driver.close()

    def _clear_database(self):
        """
        Wipes the entire database for a clean run.
        USE WITH CAUTION.
        """
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
            print("Database cleared.")

    def load_mock_data(self, json_path):
        """
        Loads triples from a JSON file into the Neo4j graph.
        This simulates the data ingestion from the Context Architect and Knowledge Engineer.
        """
        print(f"Loading mock data from {json_path}...")
        with self.driver.session() as session:
            with open(json_path, 'r') as f:
                triples = json.load(f)

            for triple in triples:
                # Extract data from the triple
                subject = triple['subject']
                obj = triple['object']
                predicate = triple['predicate']
                provenance = triple['provenance']

                # Create nodes using MERGE to avoid duplicates based on entity_id
                session.run(
                    f"MERGE (s:{subject['type']} {{id: $id, name: $name}})",
                    id=subject['entity_id'], name=subject['name']
                )
                session.run(
                    f"MERGE (o:{obj['type']} {{id: $id, name: $name}})",
                    id=obj['entity_id'], name=obj['name']
                )

                # Create the relationship with provenance properties
                query = f"""
                MATCH (s:{subject['type']} {{id: $subj_id}})
                MATCH (o:{obj['type']} {{id: $obj_id}})
                CREATE (s)-[r:{predicate} {{
                    source_text: $source_text,
                    timestamp: datetime($timestamp),
                    document_id: $document_id,
                    chapter: $chapter,
                    confidence_score: $confidence_score
                }}]->(o)
                """
                session.run(
                    query,
                    subj_id=subject['entity_id'],
                    obj_id=obj['entity_id'],
                    source_text=provenance.get('source_text'),
                    timestamp=provenance.get('timestamp'),
                    document_id=provenance.get('document_id'),
                    chapter=provenance.get('chapter'),
                    confidence_score=provenance.get('extraction_confidence')
                )
        print("Mock data loaded successfully.")

    def find_impossible_location_conflicts(self):
        """
        Runs the pre-defined Cypher query to find "impossible location" conflicts.
        """
        with self.driver.session() as session:
            results = session.run(IMPOSSIBLE_LOCATION_QUERY)
            conflicts = [record.data() for record in results]
            return conflicts

# Main execution block for the mock run
def main():
    """
    Orchestrates the mock run:
    1. Connects to Neo4j.
    2. Clears the database.
    3. Loads mock conflict data.
    4. Runs the friction detection query.
    5. Prints the results.
    """
    # Load environment variables from .env file
    load_dotenv()

    # --- Configuration ---
    # These should match your local Neo4j instance.
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    # The password is read from the NEO4J_PASSWORD environment variable.
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

    # Path to mock data
    MOCK_DATA_PATH = os.path.join("src", "mock_data", "impossible_location_conflict.json")

    print("--- Starting Friction Detector Mock Run ---")

    detector = None
    try:
        if not NEO4J_PASSWORD or NEO4J_PASSWORD == "your_actual_password":
            raise ValueError("NEO4J_PASSWORD not set correctly in .env file. Please edit the file and add your Neo4j password.")
        
        # Initialize the detector
        detector = FrictionDetector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

        # Step 1: Clear the database for a clean test
        detector._clear_database()

        # Step 2: Load the mock data that contains a known conflict
        detector.load_mock_data(MOCK_DATA_PATH)

        # Step 3: Run the detection query
        print("\nRunning friction detection query...")
        conflicts = detector.find_impossible_location_conflicts()

        # Step 4: Print the results
        if conflicts:
            print("\n--- Conflict(s) Found! ---")
            for i, conflict in enumerate(conflicts):
                print(f"\n--- Conflict {i+1} ---")
                print(f"  Character: {conflict['character_name']}")
                print(f"  Time: {conflict['conflict_time']}")
                print(f"  Location 1: {conflict['location1']} (Source: \"{conflict['source1']}\")")
                print(f"  Location 2: {conflict['location2']} (Source: \"{conflict['source2']}\")")
            print("\n--- MOCK RUN SUCCEEDED: The query correctly identified the conflict. ---")
        else:
            print("\n--- MOCK RUN FAILED: No conflicts were found. Check the query and data. ---")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please ensure your Neo4j instance is running and the credentials are correct.")

    finally:
        if detector:
            # Close the connection
            detector.close()
        print("\n--- Mock Run Complete ---")


if __name__ == "__main__":
    main()
