import spacy
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Ensure you have a spaCy model downloaded, e.g., 'en_core_web_sm'
# You would run: python -m spacy download en_core_web_sm
nlp = None

# For DeBERTa-v3 NLI Model
nli_tokenizer = None
nli_model = None

def load_spacy_model(model_name: str = "en_core_web_sm"):
    """Loads the spaCy NLP model."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(model_name)
            print(f"spaCy model '{model_name}' loaded successfully.")
        except OSError:
            print(f"spaCy model '{model_name}' not found. Please download it by running:")
            print(f"python -m spacy download {model_name}")
            raise

def load_nli_model(model_name: str = "microsoft/deberta-v3-base"):
    """Loads the DeBERTa-v3 NLI tokenizer and model."""
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None:
        print(f"Loading NLI model: {model_name}...")
        nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
        nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        nli_model.eval() # Set to evaluation mode
        print("NLI model loaded successfully.")

def extract_entities(text: str) -> list[dict]:
    """
    Identifies entities (people, organizations, locations, etc.) using spaCy's NER.
    Returns a list of dictionaries with entity text, label, start/end offsets,
    and a conceptual embedding for coreference.
    """
    if nlp is None:
        load_spacy_model()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # In a real system, you'd generate actual embeddings here (e.g., using sentenc-transformers)
        # For this conceptual implementation, we use a hash as a pseudo-embedding.
        pseudo_embedding = hash(ent.text.lower()) % 1000 # Simplified
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "pseudo_embedding": pseudo_embedding,
            "provenance": {
                "source_text_snippet": text[ent.start_char:ent.end_char+20], # Small snippet for context
                "confidence": 0.95 # Placeholder confidence for NER
            }
        })
    return entities

def extract_triples(text: str) -> list[dict]:
    """
    Extracts Subject-Predicate-Object (SPO) triples using a more robust (but still simplified)
    rule-based approach with spaCy's dependency parser.
    """
    if nlp is None:
        load_spacy_model()
    doc = nlp(text)
    triples = []
    
    # Heuristic for triple extraction (can be very complex and context-dependent)
    # This is a basic rule-based approach, not LLM-driven or exhaustive.
    for sent in doc.sents:
        # Find verbs as potential predicates
        verbs = [token for token in sent if token.pos_ == "VERB"]
        
        for verb in verbs:
            subject = None
            obj = None
            
            # Find subject (nsubj, nsubjpass, agent)
            for child in verb.children:
                if child.dep_ in ("nsubj", "nsubjpass", "agent"):
                    subject = child
                    break
            
            # Find object (dobj, pobj, attr, oprd)
            for child in verb.children:
                if child.dep_ in ("dobj", "pobj", "attr", "oprd"):
                    obj = child
                    break
            
            if subject and verb and obj:
                triples.append({
                    "subject": {"text": subject.text, "label": subject.pos_, "start_char": subject.idx, "end_char": subject.idx + len(subject.text)},
                    "predicate": {"text": verb.text, "label": verb.pos_, "start_char": verb.idx, "end_char": verb.idx + len(verb.text)},
                    "object": {"text": obj.text, "label": obj.pos_, "start_char": obj.idx, "end_char": obj.idx + len(obj.text)},
                    "provenance": {
                        "source_text_snippet": sent.text,
                        "confidence": 0.8 # Placeholder confidence for triple extraction
                    }
                })
    return triples

def resolve_coreferences(processed_chapters):
    print("--- Running simplified coreference resolution across all processed chapters ---")
    
    all_entities = []
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for entity in chunk['entities']:
                all_entities.append(entity)

    # Group entities by their pseudo_embedding and lowercased text
    # This forms the basis of our "canonical" entities
    entity_groups = {}
    for entity in all_entities:
        key = (entity['text'].lower(), entity['pseudo_embedding'])
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(entity)

    canonical_entity_mapping = {}
    canonical_id_counter = 0
    for key, group in entity_groups.items():
        # Generate a unique canonical ID for this group
        # Using a simple counter for uniqueness across groups
        canonical_id = f"CAN_ID_{canonical_id_counter}"
        canonical_id_counter += 1
        
        # Assign this canonical ID to all mentions in the group
        for entity_mention in group:
            # Use id(entity_mention) as the key to map the specific entity dictionary object
            canonical_entity_mapping[id(entity_mention)] = canonical_id
            
    print(f"--- Simplified coreference resolution complete. Found {len(entity_groups)} canonical entities ---")
    return canonical_entity_mapping


def run_nli_check(premise: str, hypothesis: str) -> dict:
    """
    Performs Natural Language Inference (NLI) using DeBERTa-v3 to determine
    if a hypothesis is entailed by, contradicted by, or neutral to a premise.
    """
    if nli_tokenizer is None or nli_model is None:
        load_nli_model()

    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # Assuming labels are 0: entailment, 1: neutral, 2: contradiction (common for NLI models)
    predicted_class_id = logits.argmax().item()
    labels = ["entailment", "neutral", "contradiction"]
    
    return {
        "prediction": labels[predicted_class_id],
        "confidence": torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    }


def process_chapters_for_nlp(chapter_data: dict) -> dict:
    """
    Processes all text chunks from all chapters through the NLP pipeline,
    including coreference resolution.

    Args:
        chapter_data (dict): Dictionary with chapter names as keys and
                             lists of text chunks (from document_parser) as values.

    Returns:
        dict: Processed chapter data with entities, triples, and canonical IDs.
    """
    if nlp is None:
        load_spacy_model()

    processed_chapters = {}
    for chapter_name, chunks in chapter_data.items():
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            entities = extract_entities(content)
            triples = extract_triples(content)

            processed_chunks.append({
                "chapter_name": chapter_name,
                "chunk_id": i,
                "content": content,
                "metadata": chunk, # Include original metadata like start_char, end_char
                "entities": entities,
                "triples": triples
            })
        processed_chapters[chapter_name] = processed_chunks

    # Perform xCoRe across all processed chapters to get a unified mapping
    canonical_entity_mapping = resolve_coreferences(processed_chapters)

    # Apply canonical IDs back to all entities and triples
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for entity in chunk['entities']:
                entity_id = id(entity) # Use the object's memory address as a temporary unique key
                if entity_id in canonical_entity_mapping:
                    entity["canonical_id"] = canonical_entity_mapping[entity_id]
                else:
                    # Fallback if an entity wasn't mapped (shouldn't happen with current logic)
                    entity["canonical_id"] = f"CAN_ENTITY_UNRESOLVED_{hash(entity['text'])}"

            for triple in chunk['triples']:
                for role in ["subject", "object"]:
                    if role in triple and "text" in triple[role]:
                        # Find the corresponding entity object to get its canonical ID
                        # This is an oversimplification; a real system would link triples to canonical entities directly
                        # based on their constituent entity mentions during graph ingestion.
                        # For now, we'll try to find a matching canonical_id from the entities in the same chunk.
                        matching_entity_canonical_id = None
                        for entity in chunk['entities']:
                            if entity['text'].lower() == triple[role]['text'].lower() and 'canonical_id' in entity:
                                matching_entity_canonical_id = entity['canonical_id']
                                break
                        triple[role]["canonical_id"] = matching_entity_canonical_id if matching_entity_canonical_id else f"CAN_ENTITY_UNRESOLVED_{hash(triple[role]['text'])}"

    return processed_chapters

if __name__ == '__main__':
    print("--- NLP Pipeline Module Test (Updated) ---")
    load_spacy_model()
    # load_nli_model() # To test NLI correction
    
    sample_chapter_data = {
        "Chapter 1": [
            {"content": "John works at Google. He lives in New York.", "document_id": "doc_ch1"},
            {"content": "Mr. Smith is a CEO. John likes apples.", "document_id": "doc_ch1"}
        ],
        "Chapter 2": [
            {"content": "The tech giant is hiring. Smith moved to California.", "document_id": "doc_ch2"},
            {"content": "New York is a bustling city. The CEO is very rich.", "document_id": "doc_ch2"}
        ]
    }

    processed_results = process_chapters_for_nlp(sample_chapter_data)

    for chapter_name, chunks in processed_results.items():
        print(f"\n--- {chapter_name} ---")
        for result in chunks:
            print(f"\nChunk ID: {result['chunk_id']}")
            print(f"Content: {result['content']}")
            print("Entities:")
            for ent in result['entities']:
                print(f"  - {ent['text']} ({ent['label']}) -> Canonical ID: {ent.get('canonical_id', 'N/A')}")
            print("Triples:")
            for triple in result['triples']:
                subj_text = triple['subject']['text']
                subj_can_id = triple['subject'].get('canonical_id', 'N/A')
                obj_text = triple['object']['text']
                obj_can_id = triple['object'].get('canonical_id', 'N/A')
                print(f"  - ({subj_text} [{subj_can_id}]) - {triple['predicate']['text']} - ({obj_text} [{obj_can_id}])")

    # Example NLI usage (uncomment load_nli_model first)
    # print("\n--- NLI Check Example ---")
    # premise = "John works at Google."
    # hypothesis = "John is employed by Google."
    # nli_result = run_nli_check(premise, hypothesis)
    # print(f"Premise: '{premise}'\nHypothesis: '{hypothesis}'\nNLI Result: {nli_result}")

    # premise = "John lives in New York."
    # hypothesis = "John lives in California."
    # nli_result = run_nli_check(premise, hypothesis)
    # print(f"Premise: '{premise}'\nHypothesis: '{hypothesis}'\nNLI Result: {nli_result}")
