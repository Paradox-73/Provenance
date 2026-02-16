# OPEN: src/nlp_pipeline.py
# PASTE the entire file content below:

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

def load_spacy_model(model_name: str = "en_core_web_trf"):
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

# --- CORRECTED MODEL NAME HERE ---
def load_nli_model(model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    """
    Loads the DeBERTa-v3 NLI tokenizer and model.
    Defaults to a pre-finetuned MNLI model to ensure the classifier head is trained.
    """
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None:
        print(f"Loading NLI model: {model_name}...")
        try:
            nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            nli_model.eval() # Set to evaluation mode
            print("NLI model loaded successfully.")
        except Exception as e:
            print(f"Error loading NLI model {model_name}: {e}")
            raise

def extract_entities(text: str) -> list[dict]:
    """
    Identifies entities (people, organizations, locations, etc.) using spaCy's NER.
    """
    if nlp is None:
        load_spacy_model()
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        # Pseudo-embedding for coreference (simplified)
        pseudo_embedding = hash(ent.text.lower()) % 1000 
        entities.append({
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "pseudo_embedding": pseudo_embedding,
            "provenance": {
                "source_text_snippet": text[ent.start_char:ent.end_char+20],
                "confidence": 0.95 
            }
        })
    return entities

def extract_triples(text: str) -> list[dict]:
    """
    Advanced Triple Extraction with Semantic Role Labeling for "Projective Locations".
    """
    if nlp is None:
        load_spacy_model()
    doc = nlp(text)
    triples = []
    
    # Verbs that indicate the *Object* is the one being located
    PROJECTIVE_VERBS = ["put", "place", "position", "locate", "spot", "see", "find"]
    
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["VERB", "AUX"]:
                
                # 1. Identify Children
                subj = [w for w in token.children if w.dep_ in ["nsubj", "nsubjpass", "csubj"]]
                obj = [w for w in token.children if w.dep_ in ["dobj", "attr", "acomp"]]
                
                # 2. Identify Prepositional Phrases
                prep_obj = []
                for child in token.children:
                    if child.dep_ == "prep": 
                        pobjs = [w for w in child.children if w.dep_ == "pobj"]
                        for po in pobjs:
                            prep_obj.append((f"{token.text} {child.text}", po))

                # --- 3. LOGIC BRANCHING ---
                
                # A. Handle "Projective Verbs" 
                # "Testimony puts YOU at the Docks"
                if token.lemma_ in PROJECTIVE_VERBS and obj and prep_obj:
                    real_subject = obj[0] 
                    for pred_text, real_location in prep_obj:
                         triples.append({
                            "subject": {"text": real_subject.text, "label": real_subject.pos_, "start_char": real_subject.idx, "end_char": real_subject.idx + len(real_subject.text)},
                            "predicate": {"text": "located at", "label": "SEMANTIC_INFERENCE", "start_char": token.idx, "end_char": real_location.idx + len(real_location.text)},
                            "object": {"text": real_location.text, "label": real_location.pos_, "start_char": real_location.idx, "end_char": real_location.idx + len(real_location.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.95}
                        })
                
                # B. Standard Extraction
                elif subj:
                    subject_node = subj[0]
                    # SVO
                    for object_node in obj:
                        triples.append({
                            "subject": {"text": subject_node.text, "label": subject_node.pos_, "start_char": subject_node.idx, "end_char": subject_node.idx + len(subject_node.text)},
                            "predicate": {"text": token.text, "label": token.pos_, "start_char": token.idx, "end_char": token.idx + len(token.text)},
                            "object": {"text": object_node.text, "label": object_node.pos_, "start_char": object_node.idx, "end_char": object_node.idx + len(object_node.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.85}
                        })
                    # Prepositional
                    for pred_text, object_node in prep_obj:
                        triples.append({
                            "subject": {"text": subject_node.text, "label": subject_node.pos_, "start_char": subject_node.idx, "end_char": subject_node.idx + len(subject_node.text)},
                            "predicate": {"text": pred_text, "label": "VERB_PHRASE", "start_char": token.idx, "end_char": object_node.idx + len(object_node.text)},
                            "object": {"text": object_node.text, "label": object_node.pos_, "start_char": object_node.idx, "end_char": object_node.idx + len(object_node.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.90}
                        })
                        
    return triples

def resolve_coreferences(processed_chapters):
    print("--- Running robust coreference resolution (Substring Matching) ---")
    all_entities = []
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for entity in chunk['entities']:
                all_entities.append(entity)

    unique_texts = list(set([e['text'] for e in all_entities]))
    unique_texts.sort(key=len, reverse=True)
    
    text_to_canonical = {}
    canonical_counter = 0

    for text in unique_texts:
        assigned = False
        for existing_text in text_to_canonical:
            if text.lower() in existing_text.lower().split():
                text_to_canonical[text] = text_to_canonical[existing_text]
                assigned = True
                break
        
        if not assigned:
            text_to_canonical[text] = f"CAN_ID_{canonical_counter}"
            canonical_counter += 1

    canonical_entity_mapping = {}
    for entity in all_entities:
        if entity['text'] in text_to_canonical:
            canonical_entity_mapping[id(entity)] = text_to_canonical[entity['text']]
            
    print(f"--- Coreference complete. Merged {len(unique_texts)} names into {canonical_counter} unique identities. ---")
    return canonical_entity_mapping

def run_nli_check(premise: str, hypothesis: str) -> dict:
    """
    Performs NLI using the loaded model.
    """
    if nli_tokenizer is None or nli_model is None:
        load_nli_model()

    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    # Mapping for MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli:
    # 0: Entailment, 1: Neutral, 2: Contradiction
    predicted_class_id = logits.argmax().item()
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    
    # Map ID to label
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    prediction = id2label.get(predicted_class_id, "neutral")
    
    return {
        "prediction": prediction,
        "confidence": confidence
    }

def process_chapters_for_nlp(chapter_data: dict) -> dict:
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
                "metadata": chunk,
                "entities": entities,
                "triples": triples
            })
        processed_chapters[chapter_name] = processed_chunks

    canonical_entity_mapping = resolve_coreferences(processed_chapters)

    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for entity in chunk['entities']:
                entity_id = id(entity)
                if entity_id in canonical_entity_mapping:
                    entity["canonical_id"] = canonical_entity_mapping[entity_id]
                else:
                    entity["canonical_id"] = f"CAN_ENTITY_UNRESOLVED_{hash(entity['text'])}"

            # Pronoun Resolution Logic
            for triple in chunk['triples']:
                for role in ["subject", "object"]:
                    role_data = triple[role]
                    role_text = role_data['text'].lower()
                    
                    if "canonical_id" in role_data and "UNRESOLVED" not in role_data["canonical_id"]:
                        continue

                    best_match_id = None
                    min_distance = float('inf')
                    
                    for entity in chunk['entities']:
                        if "canonical_id" not in entity or "UNRESOLVED" in entity["canonical_id"]:
                            continue
                            
                        dist = abs(entity['start_char'] - role_data['start_char'])
                        is_exact_match = entity['text'].lower() == role_text
                        is_pronoun_match = role_text in ["i", "he", "she", "him", "her", "you"] and entity['label'] == "PERSON"
                        
                        if is_exact_match:
                            best_match_id = entity['canonical_id']
                            break 
                        
                        if is_pronoun_match and dist < min_distance:
                            min_distance = dist
                            best_match_id = entity['canonical_id']

                    if best_match_id:
                        triple[role]["canonical_id"] = best_match_id
                    else:
                        triple[role]["canonical_id"] = f"UNRESOLVED_{hash(role_text)}"
    return processed_chapters