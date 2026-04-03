# python -m src.main chapters_data

import argparse
import os
from .neo4j_adapter import Neo4jAdapter
from .orchestrator import run_provenance_pipeline
from .nlp_pipeline import load_spacy_model, load_nli_model

def main():
    parser = argparse.ArgumentParser(description="Run the Provenance system to detect inconsistencies in literary texts.")
    parser.add_argument("input_folder", type=str,
                        help="Path to the folder containing chapter-wise .txt files.")
    parser.add_argument("--spacy_model", type=str, default="en_core_web_trf",
                        help="Name of the spaCy model to load.")
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                        help="Name of the HuggingFace NLI model.")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["spacy", "llm", "hybrid"],
                        help="NER extraction mode: 'spacy', 'llm', or 'hybrid'.")
    parser.add_argument("--ollama_model", type=str, default="qwen2.5:7b",
                        help="Local Ollama model name to use for LLM extraction.")
    parser.add_argument("--nlp_only", action="store_true",
                        help="Only run NLP processing and show the table; skip Neo4j and conflict detection.")
    
    args = parser.parse_args()

    input_folder = args.input_folder
    spacy_model = args.spacy_model
    nli_model = args.nli_model
    mode = args.mode
    ollama_model = args.ollama_model
    nlp_only = args.nlp_only

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    print(f"--- Starting Provenance Pipeline (Mode: {mode}) ---")

    # 1. Load NLP Models (Global initialization)
    try:
        load_spacy_model(spacy_model)
        load_nli_model(nli_model)
    except Exception as e:
        print(f"Failed to load NLP models: {e}")
        return

    # 2. Initialize Neo4j Adapter (Skip if nlp_only)
    neo4j_adapter = None
    if not nlp_only:
        neo4j_adapter = Neo4jAdapter()
        try:
            neo4j_adapter.connect()
            neo4j_adapter.define_schema()
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            print("Try running with --nlp_only if you just want to see the NER results.")
            return
    
    # 3. Run the Orchestrated Pipeline
    try:
        detected_conflicts = run_provenance_pipeline(
            input_folder, 
            neo4j_adapter, 
            mode=mode, 
            ollama_model=ollama_model,
            nlp_only=nlp_only
        )
        
        if not nlp_only:
            if detected_conflicts:
                print(f"--- Pipeline Completed: {len(detected_conflicts)} Inconsistencies Found ---")
            else:
                print("--- Pipeline Completed: No Inconsistencies Detected ---")
        else:
            print("--- NLP Analysis Completed ---")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
    finally:
        if neo4j_adapter:
            neo4j_adapter.close()
        print("--- Provenance Pipeline Finished ---")

if __name__ == "__main__":
    main()