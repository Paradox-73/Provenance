# Project Provenance

This project aims to process long-form narratives to find conflicting details and logical inconsistencies using an Agentic Graph-Based RAG architecture.

## Recent Updates

**Coreference Resolution Implementation:**
*   **Resolved:** The `NameError` related to `resolve_coreferences` in `src/nlp_pipeline.py` has been fixed.
*   **Implemented:** A simplified cross-document coreference resolution logic has been integrated into `src/nlp_pipeline.py`. This mechanism now groups entities across chapters based on their lowercased text and pseudo-embeddings, assigning a unique canonical ID to each. This is a foundational step towards robust entity linking.

## Current Status & Next Steps

The NLP pipeline now runs end-to-end, processing documents, extracting entities and triples, performing basic coreference resolution, and ingesting data into Neo4j. However, the system is currently reporting no conflicts, and there are warnings during Neo4j ingestion regarding unresolved subjects/objects. This indicates further refinement and implementation are needed across modules to achieve full functionality and accurate conflict detection.

The codebase is a starting point and requires dedicated effort for testing, implementing missing features, and debugging to reach a working condition as described in the `Provenance.pdf` project proposal.

---

## Team Member Task Assignments

### To All Team Members:
*   **Review `Provenance.pdf`**: Familiarize yourselves thoroughly with the "Project Provenance Implementation Plan," "Technical Role Allocation," and "Timeline and Weekly Deliverables" sections (Pages 15-17). This document is our guiding star.
*   **Review `README.md`**: Understand the latest changes and current status.
*   **Local Setup**: Ensure your development environment is correctly set up with all dependencies (Python 3.9+, spaCy, transformers, Neo4j, etc.).
*   **Communication**: Use a shared communication channel for daily stand-ups and technical discussions.

---

### Team Member 1: Context Architect
**(Focus: NLP Pipeline & Entity Linking Refinement)**

*   **Primary Files**: `src/nlp_pipeline.py`, `src/document_parser.py`
*   **Current Status**: The coreference resolution (`resolve_coreferences` in `nlp_pipeline.py`) is currently a basic entity deduplication.
*   **Tasks**:
    1.  **Refine Coreference Resolution (`src/nlp_pipeline.py`)**:
        *   **Problem**: The current `pseudo_embedding` uses a simple hash. The `Provenance.pdf` (Page 4, "Key Responsibilities: Apply semantic embeddings") explicitly mentions using "semantic embeddings (sentence-transformers)" for disambiguation.
        *   **Action**: Integrate a more sophisticated entity disambiguation using actual semantic embeddings (e.g., from `sentence-transformers` library, if available in the project, otherwise propose adding it). This will improve the accuracy of `canonical_id` assignment beyond simple text matching.
        *   **Goal**: Ensure `FR-03` ("System shall resolve coreferences across at least 10-page document spans") is actively pursued with more robust logic.
    2.  **Enhance Entity Extraction (`src/nlp_pipeline.py`)**:
        *   **Problem**: Rule-based `extract_triples` might miss complex relationships or resolve entities incorrectly, leading to "Skipping triple due to unresolved subject/object" warnings during Neo4j ingestion (as seen in `log1.txt`).
        *   **Action**: Investigate and improve the `extract_entities` and `extract_triples` functions. Focus on ensuring all relevant entities mentioned in triples are also correctly identified as standalone entities and have a `canonical_id`.
    3.  **Document Parser Robustness (`src/document_parser.py`)**:
        *   **Problem**: The `Provenance.pdf` (Page 4, "Key Responsibilities: Build robust PDF/DOCX/TXT document parsers") specifies robust parsing.
        *   **Action**: Review and enhance `document_parser.py` to ensure it reliably extracts text and metadata from various document types (PDF, DOCX, TXT) while preserving chapter/section structure. Add error handling for malformed documents.

---

### Team Member 2: Knowledge Engineer
**(Focus: Graph Construction & Data Ingestion)**

*   **Primary Files**: `src/neo4j_adapter.py`, `src/nlp_pipeline.py` (for understanding data structure)
*   **Current Status**: Neo4j ingestion shows warnings about skipping triples due to unresolved subjects/objects.
*   **Tasks**:
    1.  **Fix Triple Ingestion Logic (`src/neo4j_adapter.py`)**:
        *   **Problem**: The `log1.txt` shows "WARNING: Skipping triple due to unresolved subject/object". This is critical. The current logic to find `matching_entity_canonical_id` in `nlp_pipeline.py`'s `process_chapters_for_nlp` is an oversimplification.
        *   **Action**: In `src/neo4j_adapter.py`, when ingesting triples, ensure that both the subject and object of a triple are properly resolved to their canonical entities. If a subject/object in a triple doesn't have an existing canonical entity (or one found by the Context Architect's improved coreference), it should be created as a new entity node in Neo4j with a new canonical ID, rather than being skipped. Coordinate closely with the Context Architect on `canonical_id` generation.
        *   **Goal**: Eliminate "Skipping triple due to unresolved subject/object" warnings.
    2.  **Implement Multi-World Versioning (`src/neo4j_adapter.py`)**:
        *   **Problem**: `Provenance.pdf` (Page 5, "Key Responsibilities: Implement Multi-World Versioning") states the need to store conflicting claims as competing "realities."
        *   **Action**: Design and implement the necessary data model changes and ingestion logic in `neo4j_adapter.py` to support multi-world versioning. This will involve tagging relationships and potentially nodes with context/source information to handle conflicting assertions without overwriting.
    3.  **Enhance Triple Extraction (Coordination with Context Architect)**:
        *   **Problem**: `Provenance.pdf` (Page 5, "Key Responsibilities: Implement LLM-driven triple extraction") specifies LLM-driven extraction. Currently, it's rule-based.
        *   **Action**: Begin researching and prototyping LLM-driven triple extraction methods, potentially using fine-tuned models or few-shot prompting, to replace/augment the current rule-based `extract_triples` in `nlp_pipeline.py`. This will be an iterative process.

---

### Team Member 3: Logic Strategist
**(Focus: Conflict Detection & NLI Integration)**

*   **Primary Files**: `src/graph_logic.py`, `src/friction_queries.py`, `src/nlp_pipeline.py` (for NLI model)
*   **Current Status**: Conflict detection reports "0 conflicts" in `log1.txt`. The NLI model is loaded but not yet integrated into conflict detection.
*   **Tasks**:
    1.  **Implement Core Conflict Detection Logic (`src/graph_logic.py`, `src/friction_queries.py`)**:
        *   **Problem**: The pipeline reports 0 conflicts, and the conflict detection logic is likely rudimentary or placeholder. The `Provenance.pdf` (Page 5, "Key Responsibilities: Implement Graph-Based Friction Checking") describes this in detail.
        *   **Action**: Based on the five core contradiction types (Temporal, Inventory, Location, Identity, Attribute) outlined in `Provenance.pdf` (Page 27), start implementing the corresponding Cypher queries in `src/friction_queries.py`.
        *   **Action**: Integrate these queries into `src/graph_logic.py` to perform Graph-Based Friction Checking (`GraphFC`).
        *   **Goal**: Actively detect conflicts based on defined patterns.
    2.  **Integrate NLI for Semantic Checks (`src/graph_logic.py`, `src/nlp_pipeline.py`)**:
        *   **Problem**: The DeBERTa-v3 NLI model is loaded (`run_nli_check` in `nlp_pipeline.py`) but not used in conflict detection. `Provenance.pdf` (Page 5, "Key Responsibilities: Integrate DeBERTa-v3 NLI models") emphasizes its use.
        *   **Action**: Modify `src/graph_logic.py` to incorporate `run_nli_check` to validate semantic opposition for potential contradictions identified by Cypher queries. For example, if two statements conflict based on graph patterns, use NLI to confirm if one statement contradicts the other.
    3.  **Develop Friction Scorer (`src/graph_logic.py`)**:
        *   **Problem**: The project requires a Friction Scorer (Page 5, "Key Responsibilities: Develop the Friction Scorer").
        *   **Action**: Implement a scoring mechanism in `src/graph_logic.py` that assigns a severity and confidence score to detected conflicts, combining insights from graph patterns and NLI results.

---

### Team Member 4: System Orchestrator & Validator
**(Focus: Pipeline Flow, Testing & Reporting)**

*   **Primary Files**: `src/orchestrator.py`, `src/main.py`, new files for benchmarking/testing
*   **Current Status**: The basic pipeline runs, but no conflicts are detected, and comprehensive testing/benchmarking is absent.
*   **Tasks**:
    1.  **Refine Orchestration Flow (`src/orchestrator.py`)**:
        *   **Problem**: The current orchestration might be linear. `Provenance.pdf` (Page 6, "Key Responsibilities: Manage Agentic Orchestration via LangGraph") specifies a "Router-Planner" workflow.
        *   **Action**: Research LangGraph and integrate it into `src/orchestrator.py` to create a more dynamic, stateful workflow that can manage the data flow between extraction, ingestion, and detection stages. This includes implementing a basic "Router-Planner" loop.
    2.  **Implement Reporting (`src/orchestrator.py`)**:
        *   **Problem**: The project needs JSON/CSV reports (Page 6, "Key Responsibilities: Generate the final JSON/CSV reports").
        *   **Action**: Enhance the reporting section in `src/orchestrator.py` to generate detailed JSON/CSV reports of processed entities, triples, and, crucially, detected conflicts, adhering to the output formats described in `Provenance.pdf`.
    3.  **Develop Universal Error Injector (New Module)**:
        *   **Problem**: `Provenance.pdf` (Page 6, "Key Responsibilities: Build the Universal Error Injector") highlights the need for synthetic test cases.
        *   **Action**: Create a new Python module (e.g., `src/error_injector.py`) that can systematically inject errors (temporal, inventory, identity, etc.) into clean narrative documents to create synthetic test cases for validating conflict detection. This will be crucial for the Logic Strategist.
    4.  **Initial Benchmarking Setup**:
        *   **Problem**: `Provenance.pdf` (Page 6, "Key Responsibilities: Execute Scientific Benchmarking") requires benchmarking.
        *   **Action**: Begin setting up the framework for scientific benchmarking. This involves defining how to measure performance metrics (e.g., precision/recall for triple extraction and error detection) as described in `Provenance.pdf` (Page 30, "Success Metrics").