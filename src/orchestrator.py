from .document_parser import document_parser
from .nlp_pipeline import process_chapters_for_nlp, load_spacy_model, load_nli_model
from .neo4j_adapter import Neo4jAdapter
from .graph_logic import ConflictDetector
import os
import json
from typing import Dict, Any, List

# Define a conceptual graph state for LangGraph
# In a real LangGraph setup, this would be a TypedDict or Pydantic model
class AgenticGraphState:
    def __init__(self):
        self.raw_document_data: Dict[str, Any] = {} # Raw text chunks per chapter
        self.processed_nlp_data: Dict[str, Any] = {} # NLP processed data per chapter
        self.graph_ingestion_status: str = "pending"
        self.detected_conflicts: List[Dict[str, Any]] = []
        self.current_task: str = "start"
        self.history: List[str] = [] # To track agent decisions/actions

def node_parse_documents(state: AgenticGraphState, input_folder: str) -> AgenticGraphState:
    """
    LangGraph node: Parses documents from the input folder.
    """
    print(f"--- Orchestrator: Parsing documents from {input_folder} ---")
    chapter_data = {}
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            chapter_name = os.path.splitext(filename)[0] # Use filename as chapter name
            try:
                chunks = document_parser(file_path)
                chapter_data[chapter_name] = chunks
                state.history.append(f"Parsed chapter: {chapter_name}")
            except Exception as e:
                print(f"Error parsing {filename}: {e}")
                state.history.append(f"Error parsing {filename}: {e}")
    state.raw_document_data = chapter_data
    state.current_task = "nlp_processing"
    return state

def node_nlp_process_data(state: AgenticGraphState) -> AgenticGraphState:
    """
    LangGraph node: Processes raw document data through the NLP pipeline.
    """
    print("--- Orchestrator: Running NLP pipeline ---")
    load_spacy_model() # Ensure spaCy model is loaded
    load_nli_model()    # Ensure NLI model is loaded for semantic checks
    
    processed_data = process_chapters_for_nlp(state.raw_document_data)
    state.processed_nlp_data = processed_data
    state.current_task = "ingest_to_graph"
    state.history.append("Completed NLP processing and coreference resolution.")
    return state

def node_ingest_to_graph(state: AgenticGraphState, neo4j_adapter: Neo4jAdapter) -> AgenticGraphState:
    """
    LangGraph node: Ingests processed NLP data into the Neo4j graph.
    """
    print("--- Orchestrator: Ingesting data to Neo4j ---")
    try:
        # Save processed data for debugging
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        for chapter_name, chunks in state.processed_nlp_data.items():
            filename = os.path.join(output_dir, f"{chapter_name}_processed.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)
            print(f"Saved processed data for {chapter_name} to {filename}")

        neo4j_adapter.clear_database()
        neo4j_adapter.define_schema()
        neo4j_adapter.ingest_processed_data(state.processed_nlp_data)
        state.graph_ingestion_status = "completed"
        state.current_task = "detect_conflicts"
        state.history.append("Data successfully ingested into Neo4j.")
    except Exception as e:
        print(f"Error during Neo4j ingestion: {e}")
        state.graph_ingestion_status = "failed"
        state.current_task = "error"
        state.history.append(f"Error during Neo4j ingestion: {e}")
    return state

def node_detect_conflicts(state: AgenticGraphState, neo4j_adapter: Neo4jAdapter) -> AgenticGraphState:
    """
    LangGraph node: Detects inconsistencies using the graph_logic module.
    """
    print("--- Orchestrator: Detecting conflicts ---")
    detector = ConflictDetector(neo4j_adapter)
    conflicts = detector.detect_all_inconsistencies()
    state.detected_conflicts = conflicts
    state.current_task = "report_results"
    state.history.append(f"Conflict detection completed. Found {len(conflicts)} conflicts.")
    return state

def node_report_results(state: AgenticGraphState) -> AgenticGraphState:
    """
    LangGraph node: Finalizes and reports the detected conflicts.
    """
    print("--- Orchestrator: Reporting results ---")
    # In a real system, this would format the conflicts into a dashboard, JSON, or CSV report.
    # For now, it just prints them.
    if state.detected_conflicts:
        print("--- Final Conflict Report ---")
        for i, conflict in enumerate(state.detected_conflicts):
            print(f"Conflict {i+1} (Type: {conflict['type']}, Confidence: {conflict['confidence']:.2f}):")
            for k, v in conflict['details'].items():
                print(f"  {k}: {v}")
            print("-" * 30)
    else:
        print("No inconsistencies detected in the entire narrative.")
    
    state.current_task = "finished"
    state.history.append("Results reported.")
    return state

# Conceptual LangGraph application (simplified)
# In a real LangGraph setup, you'd define a StateGraph and add nodes/edges.
# This function simulates the execution flow.
def run_provenance_pipeline(input_folder: str, neo4j_adapter: Neo4jAdapter) -> List[Dict[str, Any]]:
    """
    Simulates the execution of the agentic provenance pipeline.
    """
    state = AgenticGraphState()

    # Step 1: Parse documents
    state = node_parse_documents(state, input_folder)
    if not state.raw_document_data:
        print("No documents found or parsed. Exiting pipeline.")
        return []

    # Step 2: NLP Process Data
    state = node_nlp_process_data(state)
    
    # Step 3: Ingest to Graph
    state = node_ingest_to_graph(state, neo4j_adapter)
    if state.graph_ingestion_status != "completed":
        print("Graph ingestion failed. Exiting pipeline.")
        return []

    # Step 4: Detect Conflicts
    state = node_detect_conflicts(state, neo4j_adapter)

    # Step 5: Report Results
    state = node_report_results(state)
    
    print("--- Pipeline Execution Summary ---")
    for step in state.history:
        print(f"- {step}")

    return state.detected_conflicts

if __name__ == '__main__':
    print("--- Orchestrator Module Conceptual Test ---")
    # This requires a dummy 'chapters' folder with .txt files and Neo4j running.
    # For a quick test, create a folder named 'chapters' in the project root
    # and put a 'Chapter 1.txt' and 'Chapter 2.txt' inside.

    dummy_chapters_folder = "chapters_test_data"
    os.makedirs(dummy_chapters_folder, exist_ok=True)
    with open(os.path.join(dummy_chapters_folder, "Chapter 1.txt"), "w") as f:
        f.write("John works at Google. He lives in New York. John has blue eyes.")
    with open(os.path.join(dummy_chapters_folder, "Chapter 2.txt"), "w") as f:
        f.write("Mr. Smith founded Google. Smith moved to California. John has brown eyes.")
    
    print(f"Created dummy chapter files in '{dummy_chapters_folder}' for testing.")

    # Initialize Neo4j Adapter (ensure .env is configured and Neo4j is running)
    neo4j_adapter = Neo4jAdapter()
    try:
        neo4j_adapter.connect()
        neo4j_adapter.define_schema() # Ensure schema is defined before ingesting

        # Run the simulated pipeline
        print("--- Running simulated provenance pipeline ---")
        conflicts = run_provenance_pipeline(dummy_chapters_folder, neo4j_adapter)
        
        if conflicts:
            print(f"Pipeline finished. Found {len(conflicts)} inconsistencies.")
        else:
            print("Pipeline finished. No inconsistencies found.")

    except Exception as e:
        print(f"Orchestrator test failed: {e}")
    finally:
        neo4j_adapter.close()
        # Clean up dummy test data
        for filename in os.listdir(dummy_chapters_folder):
            os.remove(os.path.join(dummy_chapters_folder, filename))
        os.rmdir(dummy_chapters_folder)
        print(f"Cleaned up '{dummy_chapters_folder}'.")
