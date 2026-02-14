# Kanav's Branch Update (Logic Strategist)

This branch focuses on the "Logic Strategist" role for Project Provenance. My mission is to design the algorithms that find contradictions using graph patterns and deep learning.

## What I've Done

I have implemented and tested the initial baseline for the project's core semantic logic check. This involves the `LogicValidator` class in `src/logic_baseline.py` and a new test suite `src/test_samples.py` to verify its performance.

### Baseline Model: DeBERTa for Natural Language Inference (NLI)

At the heart of our contradiction detection is a **Natural Language Inference (NLI)** model. The goal of NLI is to determine the logical relationship between two sentences: a `premise` and a `hypothesis`. The relationship can be:
1.  **Entailment**: The hypothesis is true given the premise.
2.  **Contradiction**: The hypothesis is false given the premise.
3.  **Neutral**: There is no logical relationship between the two.

We are using **DeBERTa-v3**, a state-of-the-art language model that is highly effective at understanding nuanced semantic relationships. The `logic_baseline.py` script loads a pre-trained version of this model (`cross-encoder/nli-deberta-v3-base`) to perform these checks.

### How the Code Works

1.  `logic_baseline.py`: This script defines the `LogicValidator` class. It loads the DeBERTa model and provides a `check_friction` method. This method takes a `premise` (representing an established fact from our knowledge graph) and a `hypothesis` (a new statement from the text) and uses the model to predict their relationship.
2.  `test_samples.py`: To validate this approach, I created a test suite with clear examples for each category (Contradiction, Entailment, and Neutral). The script feeds these pairs to the `LogicValidator` and checks if the model's verdict matches the expected outcome.

### Testing and Results

The test suite produced the following results, which I've captured in `log.txt`:

```
TYPE            | VERDICT         | CONFIDENCE | RESULT
------------------------------------------------------------
Contradiction   | Contradiction   | 100.0%     | PASS
Contradiction   | Contradiction   | 99.99%     | PASS
Entailment      | Entailment      | 99.7%      | PASS
Entailment      | Entailment      | 91.58%     | PASS
Neutral         | Neutral         | 99.65%     | PASS
Neutral         | Contradiction   | 99.95%     | FAIL
```

### Analysis: Why This Proves Our Project's Goal

The results are extremely revealing. The DeBERTa model is powerful and performs exceptionally well on clear-cut cases of entailment and contradiction, often with >99% confidence.

However, the **failure is the most important result**. In the last test case, the model was given two completely unrelated neutral statements:
- **Premise**: "It is raining heavily outside today."
- **Hypothesis**: "My brother just bought a new laptop."

The model incorrectly flagged this as a **Contradiction** with 99.95% confidence. This is a critical failure mode known as "hallucination," where the model creates a logical connection that does not exist.

This single failure proves the central thesis of **Project Provenance**: a pure NLI model is insufficient for reliable contradiction detection in a complex narrative. It lacks the grounding of a structured world model and can be easily confused by unrelated statements.

Our project's goal is to build a **Graph-Based RAG system** that uses a Neo4j knowledge graph as a filter. Instead of comparing every new sentence to every other sentence, we will first query the graph to see if the entities are even related. In the failed test case, our graph would show no connection between "rain" and "brother's laptop," so we would never even send that pair to the expensive and sometimes unreliable NLI model.

This test validates our hybrid approach: we will use the graph for structural and contextual logic, and the NLI model for targeted semantic checks only when the graph indicates a potential conflict. This will make our system both more accurate and more efficient.

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
