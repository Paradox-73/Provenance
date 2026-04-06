import spacy
from spacy.tokens import Doc, Span
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from tabulate import tabulate
import os
import json
import requests
import re
import difflib
from fastcoref import spacy_component

# Ensure you have a spaCy model downloaded
nlp = None
nli_tokenizer = None
nli_model = None

def load_nli_model(model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None:
        print(f"Loading NLI model: {model_name}...")
        try:
            nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            nli_model.eval()
            print("NLI model loaded successfully.")
        except Exception as e:
            print(f"Error loading NLI model {model_name}: {e}")
            raise

def run_nli_check(premise: str, hypothesis: str) -> dict:
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None: load_nli_model()

    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    confidence = torch.softmax(logits, dim=1)[0][predicted_class_id].item()
    id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    
    return {"prediction": id2label.get(predicted_class_id, "neutral"), "confidence": confidence}

class TriModeExtractor:
    def __init__(self, mode="hybrid", ollama_model="llama3.2", spacy_model="en_core_web_trf"):
        self.mode = mode
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.spacy_model_name = spacy_model
        load_spacy_model(spacy_model)
        
    def _clean_llm_json(self, raw_response):
        """
        Extracts JSON from markdown code blocks if present.
        """
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response)
        if match:
            return match.group(1).strip()
        return raw_response.strip()

    def _call_ollama(self, prompt, retries=2):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": 4096,      # Optimized for 1500-char chunks
                "num_predict": 2048,  # Finite limit prevents infinite loops
                "temperature": 0.1    # Forces strict, logical, non-creative output
            }
        }
        for attempt in range(retries):
            try:
                response = requests.post(self.ollama_url, json=payload, timeout=300)
                response.raise_for_status()
                raw_text = response.json().get("response", "")
                cleaned_text = self._clean_llm_json(raw_text)
                data = json.loads(cleaned_text)
                
                if isinstance(data, dict):
                    for key in ["entities", "triples"]:
                        if key in data and isinstance(data[key], list): 
                            return data[key]
                    for key, val in data.items():
                        if isinstance(val, list): return val
                    return [data]
                
                return data if isinstance(data, list) else []
                
            except Exception as e:
                # --- NEW: NO MORE SILENT FAILURES ---
                print(f"\n[!] Ollama attempt {attempt + 1} failed: {e}")
                if 'raw_text' in locals():
                    print(f"Raw output snippet (check if cut off): {raw_text[-200:]}\n")
                    
        return []

    def extract_entities(self, text, ledger=None):
        if self.mode == "spacy":
            ents = self._spacy_ner(text)
        elif self.mode == "llm":
            ents = self._llm_ner(text, ledger)
        elif self.mode == "hybrid":
            ents = self._hybrid_ner(text, ledger)
        else:
            ents = []
            
        valid_ents = []
        for ent in ents:
            if isinstance(ent, dict) and 'text' in ent and 'label' in ent:
                if 'start_char' not in ent: ent['start_char'] = -1
                if 'end_char' not in ent: ent['end_char'] = -1
                valid_ents.append(ent)
        return valid_ents

    def extract_triples_llm(self, text, ledger, memory=None):
        ledger_context = f"Established Characters/Items: {json.dumps(ledger)}" if ledger else ""
        memory_context = f"Current Memory State: {json.dumps(memory)}" if memory else ""
        prompt = f"""
        You are a logical auditor extracting facts for a knowledge graph.
        Extract Subject-Predicate-Object triples from the text.
        {ledger_context}
        {memory_context}
        
        CRITICAL RULES:
        1. EXHAUSTIVE EXTRACTION: You must extract EVERY relevant relationship in the text. Do not stop at just one.
        2. Use ONLY these exact predicate labels: LOCATED_AT, HAS_IDENTITY, HAS_ATTRIBUTE, POSSESSES, LOST.
        3. STRICT LOCATIONS: For LOCATED_AT, the object MUST be a geographic, architectural, or physical space (e.g., room, building, city, planet). DO NOT extract relative positions (beside, near), objects (tablet, display, console), or body parts (hand, pocket) as locations.
        4. STRICT DIRECTIONALITY: For POSSESSES and LOST, the Person MUST always be the Subject, and the Item MUST be the Object. Never make the item the subject.
        5. Resolve pronouns using the Established Characters and Memory State.
        6. KEEP IT BRIEF: The 'source_text_snippet' must be less than 10 words. Only quote the core action.
        7. STRICT ATTRIBUTES: Use HAS_ATTRIBUTE for ANY physical descriptions of a person's body, face, or condition (e.g., scars, smooth skin, eye color). NEVER use the 'LOST' predicate for physical features vanishing; instead, extract the new physical state as a new HAS_ATTRIBUTE (e.g., if a scar is missing, the attribute is 'smooth face' or 'no scar').
        
        Text: {text}
        
        Return a JSON object with two keys:
        - "triples": A JSON array of triple objects.
        - "updated_memory": An updated version of the memory state (locations of people, items held).

        Format for triples:
        [
          {{
            "subject": {{"text": "Entity Name", "canonical_id": "UPPERCASE_NAME"}},
            "predicate": {{"label": "LOCATED_AT", "text": "was at"}},
            "object": {{"text": "Location Name", "canonical_id": "UPPERCASE_LOC"}},
            "provenance": {{"source_text_snippet": "short exact quote"}}
          }}
        ]
        """
        response = self._call_ollama(prompt)
        # Handle the case where _call_ollama might return just the list or a dict
        if isinstance(response, dict):
            triples = response.get("triples", [])
            updated_memory = response.get("updated_memory", memory)
            return triples, updated_memory
        return response, memory

    def _spacy_ner(self, text):
        global nlp
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start_char": ent.start_char,
                "end_char": ent.end_char,
                "provenance": {"source": "spacy", "confidence": 0.9}
            })
        return entities

    def _llm_ner(self, text, ledger=None):
        ledger_context = f"Established Story Bible: {json.dumps(ledger)}" if ledger else ""
        prompt = f"""
        Extract all named entities (PERSON, LOC, PRODUCT) from the following text.
        {ledger_context}
        Return ONLY a JSON array of objects with 'text', 'label', 'start_char', and 'end_char' keys.
        IMPORTANT: 'start_char' and 'end_char' must be the character offsets within the provided text.
        
        Text: {text}
        """
        return self._call_ollama(prompt)

    def _hybrid_ner(self, text, ledger=None):
        spacy_ents = self._spacy_ner(text)
        ledger_context = f"Story Bible: {json.dumps(ledger)}" if ledger else ""
        prompt = f"""
        Refine these entity guesses from a text. Correct wrong labels, add missing ones.
        {ledger_context}
        
        Text: {text}
        Guesses: {json.dumps(spacy_ents)}
        
        Return ONLY a JSON array of verified objects with 'text', 'label', 'start_char', 'end_char'.
        """
        return self._call_ollama(prompt)

def load_spacy_model(model_name: str = "en_core_web_trf"):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(model_name)
            nlp.add_pipe("fastcoref")
            print(f"spaCy model '{model_name}' and fastcoref loaded.")
        except OSError:
            print(f"spaCy model '{model_name}' not found.")
            raise

def get_subject_recursive(token):
    for child in token.children:
        if child.dep_ in ["nsubj", "nsubjpass", "csubj"]:
            return child
    if token.dep_ == "xcomp" and token.head != token:
        return get_subject_recursive(token.head)
    return None

def extract_triples(text: str) -> list[dict]:
    if nlp is None: load_spacy_model()
    doc = nlp(text)
    triples = []
    PROJECTIVE_VERBS = ["put", "place", "position", "locate", "spot", "see", "find"]
    for sent in doc.sents:
        for token in sent:
            if token.pos_ in ["VERB", "AUX"]:
                subject_node = get_subject_recursive(token)
                obj = [w for w in token.children if w.dep_ in ["dobj", "attr", "acomp"]]
                prep_obj = []
                for child in token.children:
                    if child.dep_ == "prep": 
                        pobjs = [w for w in child.children if w.dep_ == "pobj"]
                        for po in pobjs:
                            prep_obj.append((f"{token.text} {child.text}", po))
                if token.lemma_ in PROJECTIVE_VERBS and obj and prep_obj:
                    real_subject = obj[0] 
                    for pred_text, real_location in prep_obj:
                         triples.append({
                            "subject": {"text": real_subject.text, "label": real_subject.pos_, "start_char": real_subject.idx, "end_char": real_subject.idx + len(real_subject.text)},
                            "predicate": {"text": "located at", "label": "SEMANTIC_INFERENCE", "start_char": token.idx, "end_char": real_location.idx + len(real_location.text)},
                            "object": {"text": real_location.text, "label": real_location.pos_, "start_char": real_location.idx, "end_char": real_location.idx + len(real_location.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.95}
                        })
                if token.lemma_ in ["have", "has", "had", "be", "is", "am", "are", "was", "were"] and subject_node and obj:
                    for object_node in obj:
                        label = "ATTRIBUTE" if token.lemma_ in ["have", "has", "had"] else "IDENTITY"
                        triples.append({
                            "subject": {"text": subject_node.text, "label": subject_node.pos_, "start_char": subject_node.idx, "end_char": subject_node.idx + len(subject_node.text)},
                            "predicate": {"text": token.text, "label": label, "start_char": token.idx, "end_char": token.idx + len(token.text)},
                            "object": {"text": object_node.text, "label": object_node.pos_, "start_char": object_node.idx, "end_char": object_node.idx + len(object_node.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.90}
                        })
                if subject_node:
                    for object_node in obj:
                        triples.append({
                            "subject": {"text": subject_node.text, "label": subject_node.pos_, "start_char": subject_node.idx, "end_char": subject_node.idx + len(subject_node.text)},
                            "predicate": {"text": token.text, "label": token.pos_, "start_char": token.idx, "end_char": token.idx + len(token.text)},
                            "object": {"text": object_node.text, "label": object_node.pos_, "start_char": object_node.idx, "end_char": object_node.idx + len(object_node.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.85}
                        })
                    for pred_text, object_node in prep_obj:
                        triples.append({
                            "subject": {"text": subject_node.text, "label": subject_node.pos_, "start_char": subject_node.idx, "end_char": subject_node.idx + len(subject_node.text)},
                            "predicate": {"text": pred_text, "label": "VERB_PHRASE", "start_char": token.idx, "end_char": object_node.idx + len(object_node.text)},
                            "object": {"text": object_node.text, "label": object_node.pos_, "start_char": object_node.idx, "end_char": object_node.idx + len(object_node.text)},
                            "provenance": {"source_text_snippet": sent.text, "confidence": 0.90}
                        })
    return triples

def resolve_coreferences_neural(text):
    global nlp
    if nlp is None: load_spacy_model()
    doc = nlp(text, component_cfg={"fastcoref": {"resolve_text": True}})
    return doc._.resolved_text, doc._.coref_clusters

def process_chapters_for_nlp(chapter_data: dict, mode="hybrid", ollama_model="qwen2.5:7b", ledger=None) -> dict:
    if nlp is None: load_spacy_model()
    extractor = TriModeExtractor(mode=mode, ollama_model=ollama_model)
    if ledger is None: ledger = {}

    processed_chapters = {}
    current_memory = {} # Initialize memory state for stateful extraction

    for chapter_name, chunks in chapter_data.items():
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            resolved_content, clusters = resolve_coreferences_neural(content)
            
            # NER on RESOLVED content
            entities = extractor.extract_entities(resolved_content, ledger)
            for ent in entities: ledger[ent['text']] = ent['label']

            # Triple Extraction (Phase 2) - STATEFUL
            all_chunk_triples = []
            if mode in ["llm", "hybrid"]:
                doc = nlp(resolved_content)
                sentences = [sent.text for sent in doc.sents]
                
                # Process in windows of 2 sentences for context + memory
                for j in range(0, len(sentences), 2):
                    window_text = " ".join(sentences[j:j+2])
                    triples, updated_memory = extractor.extract_triples_llm(window_text, ledger, current_memory)
                    if isinstance(triples, list):
                        all_chunk_triples.extend(triples)
                    current_memory = updated_memory
            else:
                all_chunk_triples = extract_triples(resolved_content)

            processed_chunks.append({
                "chapter_name": chapter_name,
                "chunk_id": i,
                "content": content,
                "resolved_content": resolved_content,
                "entities": entities,
                "triples": all_chunk_triples,
                "coref_clusters": clusters
            })
        processed_chapters[chapter_name] = processed_chunks
    
    # Smarter Canonical IDs (Phase 1.2)
    known_ids = list(ledger.keys())
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for ent in chunk['entities']:
                raw_name = ent['text']
                
                # --- NEW: Robust Fuzzy Deduplication ---
                best_match = None
                matches = difflib.get_close_matches(raw_name, known_ids, n=1, cutoff=0.8)
                if matches:
                    best_match = matches[0]
                else:
                    # Fallback to containment
                    for existing_name in known_ids:
                        if raw_name != existing_name:
                            if raw_name in existing_name or existing_name in raw_name:
                                best_match = existing_name
                                break
                
                resolved_name = best_match if best_match else raw_name
                if not best_match and resolved_name not in known_ids:
                    known_ids.append(resolved_name)
                
                if ent['label'] in ['PERSON', 'TITLE']:
                    ent['canonical_id'] = resolved_name.split()[-1].upper()
                else:
                    ent['canonical_id'] = resolved_name.upper().replace(' ', '_')
            
            for triple in chunk['triples']:
                for role in ['subject', 'object']:
                    node = triple[role]
                    if not isinstance(node, dict):
                        triple[role] = {"text": node}
                        node = triple[role]
                    
                    text = node.get('text', '')
                    
                    # Deduplicate triple nodes too
                    best_match = None
                    matches = difflib.get_close_matches(text, known_ids, n=1, cutoff=0.8)
                    if matches:
                        best_match = matches[0]
                    else:
                        for existing_name in known_ids:
                            if text != existing_name:
                                if text in existing_name or existing_name in text:
                                    best_match = existing_name
                                    break
                    
                    resolved_text = best_match if best_match else text
                    if not best_match and resolved_text not in known_ids:
                        known_ids.append(resolved_text)
                    
                    label = "UNKNOWN"
                    for e in chunk['entities']:
                        if e['text'] == text:
                            label = e['label']
                            break
                    
                    if label in ['PERSON', 'TITLE']:
                        triple[role]['canonical_id'] = resolved_text.split()[-1].upper()
                    else:
                        triple[role]['canonical_id'] = resolved_text.upper().replace(' ', '_')

    return processed_chapters, ledger

def print_ner_table(processed_chapters):
    data = []
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            resolved_text = chunk.get("resolved_content", chunk["content"])
            doc = nlp(resolved_text)
            sents = list(doc.sents)
            for ent in chunk['entities']:
                start = ent.get('start_char', -1)
                
                sent_text = "Unknown"
                sent_no = -1
                
                if start != -1:
                    for idx, sent in enumerate(sents):
                        if start >= sent.start_char and start < sent.end_char:
                            sent_text = sent.text.strip()
                            sent_no = idx + 1
                            break
                data.append({
                    "Chapter": chapter_name, "Sent #": sent_no, "Text": ent['text'],
                    "Label": ent['label'], "Resolved ID": ent.get('canonical_id', '???'),
                    "Sentence Snippet": (sent_text[:60] + '...') if len(sent_text) > 60 else sent_text
                })
    df = pd.DataFrame(data)
    if not df.empty:
        print("\n--- NER and Coreference Resolution Report ---")
        print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))
    else:
        print("\nNo entities found for the report.")

if __name__ == "__main__":
    chapter_file = os.path.join("chapters_data", "Chapter 1.txt")
    if os.path.exists(chapter_file):
        with open(chapter_file, "r") as f: text = f.read()
        sample_data = {"Chapter 1": [{"content": text}]}
        results, _ = process_chapters_for_nlp(sample_data, mode="spacy")
        print_ner_table(results)
