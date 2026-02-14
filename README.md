# Kanav's Branch Update (Logic Strategist)

This branch focuses on the "Logic Strategist" role for Project Provenance. My mission is to design the algorithms that find contradictions using graph patterns and deep learning.

## What I've Done

I have implemented the initial baseline for logical friction detection in `src/logic_baseline.py`.

This script contains the `LogicValidator` class, which utilizes a pre-trained DeBERTa-v3 model from Hugging Face (`cross-encoder/nli-deberta-v3-base`). This class provides a `check_friction` method that takes a `premise` (the existing context from the graph) and a `hypothesis` (a new assertion) and determines if they are a "Contradiction", "Entailment", or "Neutral".

This is the core of our semantic entailment and contradiction checking, as outlined in the project plan.

## What I Need From The Team

To proceed with building and testing the full friction detection engine, I need the following from other team members:

### From Member 1 (Context Architect)

You are working on the NLP pipeline to parse the text. To ensure the data can be correctly consumed by the logic and graph modules, please ensure your output JSON for each extracted triple follows this exact structure:

```json
{
  "extraction_id": "<unique_id>",
  "subject": {
    "entity_id": "<entity_id>",
    "name": "<entity_name>",
    "type": "<entity_type>"
  },
  "predicate": "<relationship>",
  "object": {
    "entity_id": "<entity_id>",
    "name": "<entity_name>",
    "type": "<entity_type>"
  },
  "provenance": {
    "source_text": "<original_text_snippet>",
    "document_id": "<doc_id>",
    "chapter": "<chapter_num>",
    "sequence_id": "<int_sequence>",
    "extraction_confidence": "<float_score>"
  }
}
```

### From Member 2 (Knowledge Engineer)

You are responsible for the Neo4j graph structure. Please implement a Labeled Property Graph (LPG) schema that looks exactly like this. This structure is critical for my Cypher queries to function correctly.

**Node Labels:**
*   `Character` (Properties: `id`, `name`, `aliases`)
*   `Location` (Properties: `id`, `name`, `type`)
*   `Object` (Properties: `id`, `name`, `description`)
*   `Event` (Properties: `id`, `name`, `chapter`)
*   `Trait` (Properties: `id`, `description`, `category`)
*   `Rule` (Properties: `id`, `description`, `domain`)

**Edge Types (Relationships):**
*Every single edge* must contain the following provenance metadata: `source_text`, `timestamp` (or chronological sequence ID), `chapter_id`, and `confidence_score`.

*   `[:LOCATED_AT]` (Links Character/Object to Location)
*   `[:OWNS]` / `[:POSSESSES]` (Links Character to Object)
*   `[:HAS_ATTRIBUTE]` (Links Character to a physical Trait)
*   `[:EXHIBITS_TRAIT]` (Links Character to a psychological Trait)
*   `[:BOUND_BY]` (Links Character/Event to a Rule)
*   `[:PARTICIPATES_IN]` (Links Character to an Event)
