from .document_parser import document_parser
from .nlp_pipeline import process_chapters_for_nlp, load_spacy_model, load_nli_model, print_ner_table
from .neo4j_adapter import Neo4jAdapter
from .graph_logic import ConflictDetector
import os
import json
from typing import Dict, Any, List

class AgenticGraphState:
    def __init__(self):
        self.raw_document_data: Dict[str, Any] = {} 
        self.processed_nlp_data: Dict[str, Any] = {} 
        self.entity_ledger: Dict[str, str] = {} 
        self.graph_ingestion_status: str = "pending"
        self.detected_conflicts: List[Dict[str, Any]] = []
        self.current_task: str = "start"
        self.history: List[str] = [] 

def node_parse_documents(state: AgenticGraphState, input_folder: str) -> AgenticGraphState:
    print(f"--- Orchestrator: Parsing documents from {input_folder} ---")
    chapter_data = {}
    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".txt"):
            file_path = os.path.join(input_folder, filename)
            chapter_name = os.path.splitext(filename)[0] 
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

def node_nlp_process_data(state: AgenticGraphState, mode="hybrid", ollama_model="qwen2.5:7b") -> AgenticGraphState:
    print(f"--- Orchestrator: Running NLP pipeline (Mode: {mode}) ---")
    load_spacy_model() 
    load_nli_model()    
    
    processed_data, updated_ledger = process_chapters_for_nlp(
        state.raw_document_data, 
        mode=mode, 
        ollama_model=ollama_model, 
        ledger=state.entity_ledger
    )
    
    print_ner_table(processed_data)
    
    state.processed_nlp_data = processed_data
    state.entity_ledger = updated_ledger
    state.current_task = "ingest_to_graph"
    state.history.append(f"Completed NLP processing with mode '{mode}'. Story Bible has {len(updated_ledger)} entries.")
    return state

def node_ingest_to_graph(state: AgenticGraphState, neo4j_adapter: Neo4jAdapter) -> AgenticGraphState:
    print("--- Orchestrator: Ingesting data to Neo4j ---")
    try:
        output_dir = "processed_data"
        os.makedirs(output_dir, exist_ok=True)
        for chapter_name, chunks in state.processed_nlp_data.items():
            filename = os.path.join(output_dir, f"{chapter_name}_processed.json")
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, indent=2, ensure_ascii=False)

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
    print("--- Orchestrator: Detecting conflicts ---")
    detector = ConflictDetector(neo4j_adapter)
    conflicts = detector.detect_all_inconsistencies()
    state.detected_conflicts = conflicts
    state.current_task = "report_results"
    state.history.append(f"Conflict detection completed. Found {len(conflicts)} conflicts.")
    return state

def node_report_results(state: AgenticGraphState) -> AgenticGraphState:
    print("--- Orchestrator: Reporting results ---")
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

def run_provenance_pipeline(input_folder: str, neo4j_adapter: Neo4jAdapter = None, mode="hybrid", ollama_model="qwen2.5:7b", nlp_only=False) -> List[Dict[str, Any]]:
    state = AgenticGraphState()

    state = node_parse_documents(state, input_folder)
    if not state.raw_document_data:
        print("No documents found or parsed. Exiting pipeline.")
        return []

    state = node_nlp_process_data(state, mode=mode, ollama_model=ollama_model)

    if nlp_only:
        print("--- Pipeline stopping after NLP (nlp_only=True) ---")
        return []

    if neo4j_adapter:
        state = node_ingest_to_graph(state, neo4j_adapter)
        if state.graph_ingestion_status != "completed":
            print("Graph ingestion failed. Exiting pipeline.")
            return []
    else:
        print("No Neo4j adapter provided. Skipping ingestion.")
        return []

    state = node_detect_conflicts(state, neo4j_adapter)
    state = node_report_results(state)
    
    print("--- Pipeline Execution Summary ---")
    for step in state.history:
        print(f"- {step}")

    return state.detected_conflicts

if __name__ == '__main__':
    print("--- Orchestrator Module Conceptual Test ---")
    dummy_chapters_folder = "chapters_test_data"
    os.makedirs(dummy_chapters_folder, exist_ok=True)
    with open(os.path.join(dummy_chapters_folder, "Chapter 1.txt"), "w") as f:
        f.write("John works at Google. He lives in New York. John has blue eyes.")
    
    neo4j_adapter = Neo4jAdapter()
    try:
        neo4j_adapter.connect()
        neo4j_adapter.define_schema()
        conflicts = run_provenance_pipeline(dummy_chapters_folder, neo4j_adapter)
    except Exception as e:
        print(f"Orchestrator test failed: {e}")
    finally:
        neo4j_adapter.close()
        if os.path.exists(dummy_chapters_folder):
            for filename in os.listdir(dummy_chapters_folder):
                os.remove(os.path.join(dummy_chapters_folder, filename))
            os.rmdir(dummy_chapters_folder)
