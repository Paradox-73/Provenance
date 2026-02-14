# Project Provenance

An agentic system for tracking narrative continuity and detecting logical friction in long-form stories. It helps authors find plot holes and maintain consistency by building a "World State" from their text.

This project processes narratives to find conflicting details and logical inconsistencies. While the architecture can be applied to other domains like legal analysis, our current focus is on **Creative Narrative Continuity**.

## Key Features

*   **Automated World-State Tracking**: Extracts and tracks entities like characters, locations, objects, and their states across a narrative.
*   **Continuity Error Detection**: Automatically flags continuity errors (e.g., a character being in two places at once, an object appearing from nowhere, inconsistent physical attributes).
*   **Deep Provenance Graph**: Uses a Neo4j graph to store every piece of information with metadata about its origin (source text, chapter, page), creating a fully auditable chain of facts.
*   **Agentic Auditing**: An intelligent agent explores the text and the knowledge graph to discover and report on logical inconsistencies.
*   **Conflict Dashboard**: A planned UI to visualize detected conflicts and allow authors to trace them back to the specific lines in the source text.

## Tech Stack

*   **Backend**: Python
*   **Graph Database**: Neo4j
*   **NLP**: spaCy, Hugging Face Transformers (DeBERTa-v3 for NLI)
*   **Orchestration**: LangGraph

## How It Works

The system uses a three-tier agentic architecture to analyze a narrative:

1.  **Text Ingestion & NLP**: The system parses documents (PDF, TXT, DOCX) and uses an NLP pipeline to identify entities (characters, locations) and the relationships between them.
2.  **Graph Construction**: This information is used to build a rich knowledge graph in Neo4j. Every fact and relationship is stored with "provenance"—metadata linking it back to the exact source text. Conflicting claims are not overwritten but stored as competing "realities."
3.  **Friction Detection**: An agent-driven workflow queries the graph to find structural and semantic contradictions. This allows it to detect inconsistencies that a simple text search would miss.

## Getting Started

```bash
# Clone the repository
git clone <repository-url>
cd Provenance

# Install dependencies
pip install -r requirements.txt

# Further setup for Neo4j and other services will be added here.
```
