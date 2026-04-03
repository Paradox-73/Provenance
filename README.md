# Named Entity Recognition (NER) & NLP Implementation in Provenance

This document provides a technical overview of the Named Entity Recognition (NER) and Natural Language Processing (NLP) architecture implemented in the Provenance project.

## 1. Core Libraries and Models
The system integrates several advanced NLP libraries and models to achieve high-precision extraction and logical auditing:
- **spaCy (v3.7.0):** The primary NLP framework, using the `en_core_web_trf` transformer model for NER and dependency parsing.
- **fastcoref:** Integrated as a spaCy component for neural coreference resolution.
- **HuggingFace Transformers:** Utilizes `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (or `cross-encoder/nli-deberta-v3-base`) for Natural Language Inference (NLI).
- **Ollama:** Supports local LLM-based extraction (e.g., `qwen2.5:7b`, `llama3.2`) for complex semantic tasks.
- **PyTorch:** Backend for transformer-based models.
- **Neo4j:** Graph database for storing entities, relationships, and assertions.

## 2. Pipeline Architecture
The system follows a modular "Research -> Strategy -> Execution" flow managed by an agentic state pattern.

### 2.1 Orchestration & Entry Point
- **`src/main.py`:** The CLI entry point. It handles argument parsing, initializes global NLP models (spaCy, NLI), and triggers the orchestrated pipeline.
- **`src/orchestrator.py`:** Implements `AgenticGraphState` to manage document data, NLP results, and the "Story Bible" (entity ledger). It coordinates the transition between parsing, NLP processing, graph ingestion, and conflict detection.

### 2.2 Data Ingestion & Parsing
- **`src/document_parser.py`:** Extracts text from `.pdf`, `.docx`, and `.txt` files. It implements overlapping chunking (default: 3000 chars with 300 overlap) to preserve context across boundaries.

### 2.3 NLP Pipeline (`src/nlp_pipeline.py`)
This is the core processing engine, supporting three extraction modes:
1.  **`spacy` Mode:** Uses standard transformer-based NER and custom dependency-tree triple extraction.
2.  **`llm` Mode:** Delegates extraction to a local Ollama model using structured JSON prompting.
3.  **`hybrid` Mode (Default):** Uses spaCy for initial guesses and an LLM for refinement and complex relationship extraction.

**Key Features:**
- **Neural Coreference Resolution:** Resolves pronouns (e.g., "he", "she", "it") to their original entities before extraction.
- **Triple Extraction:** Identifies Subject-Predicate-Object (SVO) relationships. It specifically handles "Projective Verbs" (e.g., *put, place, find*) to infer location and movement.
- **Canonical ID Generation:** Implements entity deduplication and normalization (e.g., mapping "Kael" and "Commander Kael" to the same ID).
- **NLI Integration:** Provides the `run_nli_check` function for semantic verification of statements.

### 2.4 Graph Integration (`src/neo4j_adapter.py`)
- Manages the connection to Neo4j and defines the graph schema (constraints and indexes).
- Implements a **two-pass ingestion logic**:
    - **Pass 1:** Creates nodes for `Person`, `Location`, and general `Entity`.
    - **Pass 2:** Maps extracted triples to specific graph relationships: `LOCATED_AT`, `HAS_IDENTITY`, `HAS_ATTRIBUTE`, or generic `HAS_ASSERTION` nodes for complex predicates.

### 2.5 Conflict Detection & Logic
- **`src/graph_logic.py`:** Contains the `ConflictDetector` class. It uses Cypher queries to find potential contradictions and validates them using NLI:
    - **Temporal/Location Conflicts:** Detecting if a person is in two places at once.
    - **Inventory Conflicts:** Tracking POSSESSION vs. LOSS of items.
    - **Identity/Attribute Conflicts:** Identifying contradictory descriptions (e.g., "jagged scar" vs. "smooth face").
- **`src/friction_queries.py`:** Stores centralized Cypher queries used by the detector.
- **`src/logic_baseline.py`:** Provides a standalone `LogicValidator` for direct NLI checks between two narrative strings.

## 3. Semantic Classification
The system uses NLI to categorize predicates dynamically. For example, `_classify_predicate` in `src/graph_logic.py` checks if a verb like "dropped" or "lost" entails a "LOSS" of possession, allowing for sophisticated inventory tracking without hardcoded verb lists.

## 4. Current Workflow Status
The implementation has transitioned from a pure spaCy-based system to a **Hybrid LLM-Graph architecture**. This allows the system to handle the nuance of literary text (metaphors, indirect speech) while maintaining the structural rigor of a knowledge graph.
