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
                        help="Name of the spaCy model to load (e.g., 'en_core_web_sm', 'en_core_web_lg', 'en_core_web_trf').")
    # --- CORRECTED MODEL NAME HERE ---
    parser.add_argument("--nli_model", type=str, default="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
                        help="Name of the HuggingFace NLI model (must be fine-tuned on MNLI).")
    
    args = parser.parse_args()

    input_folder = args.input_folder
    spacy_model = args.spacy_model
    nli_model = args.nli_model

    if not os.path.isdir(input_folder):
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return

    print(f"--- Starting Provenance Pipeline for folder: {input_folder} ---")

    # 1. Load NLP Models (Global initialization)
    try:
        load_spacy_model(spacy_model)
        load_nli_model(nli_model)
    except Exception as e:
        print(f"Failed to load NLP models: {e}")
        print("Please ensure spaCy models are downloaded ('python -m spacy download en_core_web_trf')")
        print("and HuggingFace models are accessible.")
        return

    # 2. Initialize Neo4j Adapter
    neo4j_adapter = Neo4jAdapter()
    try:
        neo4j_adapter.connect()
        neo4j_adapter.define_schema() # Ensure schema is defined before any ingestion attempt
    except Exception as e:
        print(f"Failed to connect to or initialize Neo4j: {e}")
        print("Please ensure your Neo4j database is running and credentials in .env are correct.")
        return
    
    # 3. Run the Orchestrated Pipeline
    try:
        detected_conflicts = run_provenance_pipeline(input_folder, neo4j_adapter)
        
        if detected_conflicts:
            print(f"--- Pipeline Completed: {len(detected_conflicts)} Inconsistencies Found ---")
        else:
            print("--- Pipeline Completed: No Inconsistencies Detected ---")

    except Exception as e:
        print(f"An error occurred during pipeline execution: {e}")
    finally:
        # 4. Close Neo4j Connection
        neo4j_adapter.close()
        print("--- Provenance Pipeline Finished ---")

if __name__ == "__main__":
    main()