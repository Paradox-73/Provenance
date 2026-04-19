import spacy
from spacy.tokens import Doc, Span, Token
from spacy.matcher import Matcher
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

# Global spaCy model
nlp = None
nli_tokenizer = None
nli_model = None

def load_nli_model(model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"):
    global nli_tokenizer, nli_model
    if nli_tokenizer is None or nli_model is None:
        try:
            nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            nli_model.eval()
        except Exception as e:
            print(f"Error loading NLI model: {e}")
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

def load_spacy_model(model_name: str = "en_core_web_trf", ledger: dict = None):
    global nlp
    if nlp is None:
        try:
            print(f"Loading spaCy model: {model_name}...")
            nlp = spacy.load(model_name)
            # Add coref to the end of the pipeline
            nlp.add_pipe("fastcoref")
            if ledger:
                print(f"Adding entity ruler with {len(ledger)} patterns...")
                ruler = nlp.add_pipe("entity_ruler", before="ner")
                patterns = [{"label": label, "pattern": text} for text, label in ledger.items() if label]
                ruler.add_patterns(patterns)
        except Exception as e:
            print(f"Error loading spaCy model: {e}")
            raise

def merge_titles_in_ents(doc):
    """Manually merges titles like 'Captain' into adjacent PERSON entities."""
    titles = ["Captain", "Commander", "Dr.", "Doctor", "Professor", "Chief", "Officer", "Lieutenant"]
    new_ents = []
    ents = list(doc.ents)
    i = 0
    while i < len(ents):
        ent = ents[i]
        # Check if previous word was a title
        start_token_idx = ent.start
        if start_token_idx > 0:
            prev_token = doc[start_token_idx - 1]
            if prev_token.text in titles:
                # Create a new Span that includes the title
                new_ent = Span(doc, start_token_idx - 1, ent.end, label=ent.label)
                new_ents.append(new_ent)
                i += 1
                continue
        new_ents.append(ent)
        i += 1
    
    # Filter out overlapping entities (prefer longer ones)
    try:
        doc.ents = spacy.util.filter_spans(new_ents)
    except:
        pass
    return doc

class SemanticTranslator:
    def __init__(self, ollama_model="qwen2.5:7b"):
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        self.cache = {}

    def translate_predicate(self, verb_text, subject_text, object_text):
        """Maps a raw action verb to a core narrative predicate."""
        verb_lower = verb_text.lower()
        
        # FAST PATH: Rule-based mapping for 90% of cases
        if verb_lower in ["is", "am", "are", "was", "were", "become", "became"]:
            return "HAS_ATTRIBUTE" if len(object_text.split()) < 3 else "HAS_IDENTITY"
        if verb_lower in ["live", "lives", "lived", "stay", "stays", "stayed", "stand", "stood", "enter", "entered"]:
            return "LOCATED_AT"
        if verb_lower in ["has", "have", "had", "possess", "possesses", "hold", "holding", "held", "carry", "carrying"]:
            return "POSSESSES"
        if verb_lower in ["lose", "lost", "drop", "dropped", "leave", "left"]:
            return "LOST"

        # SLOW PATH: Only call LLM if rules fail
        cache_key = f"{verb_text}_{object_text}"
        if cache_key in self.cache: return self.cache[cache_key]
        # ... (rest of LLM logic)

        prompt = f"""Analyze the action: "{subject_text} {verb_text} {object_text}".
Map this action to EXACTLY ONE of these core predicates:
- LOCATED_AT (movement, entering, standing in a place, living in a place)
- POSSESSES (holding, taking, owning, gaining an item)
- LOST (dropping, breaking, leaving, losing an item)
- HAS_ATTRIBUTE (describing a physical trait, appearance, or simple state)
- HAS_IDENTITY (defining a role, job, or name)
- NONE (if it is a general action or emotion)

Return ONLY the predicate name in uppercase.
"""
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            res = response.json().get("response", "NONE").strip().upper()
            # Clean up potential LLM chatter
            for p in ["LOCATED_AT", "POSSESSES", "LOST", "HAS_ATTRIBUTE", "HAS_IDENTITY"]:
                if p in res:
                    self.cache[cache_key] = p
                    return p
        except: pass
        return "NONE"

    def resolve_alias(self, entity_name, existing_ledger):
        """Checks if a new entity name is an alias of an existing one."""
        if not entity_name or not isinstance(entity_name, str): return entity_name
        
        # 1. JUNK & PRONOUN FILTERING
        clean_name = entity_name.strip(' .,:;"\'”’!?')
        pronouns = ["I", "me", "my", "he", "him", "his", "she", "her", "it", "its", "they", "them", "their", "we", "us", "our"]
        if clean_name.lower() in pronouns or len(clean_name) < 2:
            return None # Signal to skip this as a node

        if not clean_name[0].isupper() or not any(c.isalpha() for c in clean_name):
            return clean_name

        # 2. CACHING
        if hasattr(self, 'alias_cache') and clean_name in self.alias_cache:
            return self.alias_cache[clean_name]
        if not hasattr(self, 'alias_cache'): self.alias_cache = {}

        if not existing_ledger: return clean_name
        
        candidates = [name for name, label in existing_ledger.items() if label in ['PERSON', 'TITLE', 'ORG']]
        if not candidates: return clean_name

        # 3. STRICT PROMPT
        prompt = f"""Identify if this entity is an alias of an existing one.
ENTITY: "{clean_name}"
EXISTING: {candidates[:20]}

If it matches one, return that name.
If it is new, return "{clean_name}".
Response format: {{"resolved_name": "actual_name_string"}}
"""
        payload = {"model": self.ollama_model, "prompt": prompt, "stream": False, "format": "json"}
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=10)
            res_data = json.loads(response.json().get("response", "{}"))
            resolved = res_data.get("resolved_name", clean_name).strip()
            if "NEW NAME" in resolved or "actual_name" in resolved: resolved = clean_name
            self.alias_cache[clean_name] = resolved
            return resolved
        except: return clean_name

translator = SemanticTranslator()

def extract_triples_deterministic(doc: Doc) -> list[dict]:
    # This function is kept for backward compatibility or direct use
    # but the logic is now mostly integrated into process_chapters_for_nlp
    return []

class TriModeExtractor:
    def __init__(self, mode="hybrid", ollama_model="qwen2.5:7b"):
        self.mode = mode
        self.ollama_model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"

    def _clean_llm_json(self, raw_response):
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_response)
        if match:
            return match.group(1).strip()
        return raw_response.strip()

    def _call_ollama(self, system_prompt, text_chunk, retries=2):
        payload = {
            "model": self.ollama_model,
            "prompt": f"Text to analyze:\n{text_chunk}",
            "system": system_prompt,
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.1
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
                    for key, val in data.items():
                        if isinstance(val, list):
                            data = val
                            break
                    if isinstance(data, dict):
                        data = [data]
                return data if isinstance(data, list) else []
                
            except Exception as e:
                print(f"[!] Ollama attempt {attempt + 1} failed: {e}")
        return []

    def extract_triples_llm(self, text, raw_text_chunk, canonical_entities=None):
        if canonical_entities is None: canonical_entities = []
        system_prompt = f"""You are a strict Narrative State Tracker. Analyze the text chunk and extract literal, physical state changes AND static character facts (identities, jobs, physical attributes).

CRITICAL RULES:
1. EXHAUSTIVE EXTRACTION: Extract EVERY relevant relationship. Do not stop at just one. Find every instance of someone being located somewhere, possessing something, losing something, or having an attribute/identity.
2. METAPHOR FILTER: If the text uses a metaphor (e.g., 'she was in a dark place mentally', 'he lost his mind'), set 'is_metaphor' to true.
3. EDGE CASES: Capture 'LOST' for dropping, breaking, or forgetting items. Capture 'LOCATED_AT' for entering rooms or standing in areas.
4. PRONOUN RESOLUTION: If the text is dialogue or uses first-person (e.g., 'I have the key'), you MUST resolve 'I', 'me', or 'my' to the name of the character speaking.
5. CANONICAL NAMING: Use these exact entity names if they match: {canonical_entities}.

Output ONLY a JSON array of objects:
{{
  "character_id": "string",
  "state_change_type": "ENUM: [LOCATED_AT, POSSESSES, LOST, HAS_ATTRIBUTE, HAS_IDENTITY]",
  "target_value": "string",
  "is_metaphor": "boolean",
  "timestamp": "string (e.g., '12:00' or 'unknown')",
  "target_feature": "string (e.g., 'scar' or 'general')"
}}"""
        raw_results = self._call_ollama(system_prompt, text)
        
        triples = []
        for item in raw_results:
            is_metaphor_raw = item.get("is_metaphor", False)
            is_metaphor = str(is_metaphor_raw).lower() == "true"
            
            if not is_metaphor:
                ts_raw = item.get("timestamp")
                ts = str(ts_raw) if ts_raw and str(ts_raw).lower() not in ["null", "none", "", "unknown"] else "unknown"
                tf_raw = item.get("target_feature")
                tf = str(tf_raw) if tf_raw and str(tf_raw).lower() not in ["null", "none", "", "general"] else "general"
                sc_type = str(item.get("state_change_type", "UNKNOWN")).upper().replace("ENUM:", "").replace("[", "").replace("]", "").strip()
                t_val = str(item.get("target_value", "UNKNOWN"))
                
                triples.append({
                    "subject": {"text": str(item.get("character_id", "UNKNOWN"))},
                    "predicate": {
                        "label": sc_type, 
                        "text": sc_type + "_" + t_val
                    },
                    "object": {"text": t_val},
                    "timestamp": ts,
                    "target_feature": tf,
                    "provenance": {"source_text_snippet": raw_text_chunk}
                })
        return triples

def process_chapters_for_nlp(chapter_data: dict, mode="hybrid", ollama_model="qwen2.5:7b", ledger=None) -> tuple:
    if nlp is None: load_spacy_model(ledger=ledger)
    if ledger is None: ledger = {}
    processed_chapters = {}
    
    extractor = TriModeExtractor(mode=mode, ollama_model=ollama_model)
    
    def get_canonical_id(text):
        if not text: return "UNKNOWN"
        cid = re.sub(r'[^A-Z0-9_]', '', str(text).upper().replace(' ', '_'))
        return cid if cid else "UNKNOWN"

    for chapter_name, chunks in chapter_data.items():
        processed_chunks = []
        last_seen_person = None 
        
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            # Process with coref
            doc = nlp(content)
            
            # NER can fail on large chunks with 'trf' models due to token limits.
            # Fallback: if doc.ents is empty, process sentence by sentence.
            entities_raw = list(doc.ents)
            if not entities_raw:
                print(f"  [NLP] Warning: No entities in {chapter_name} chunk {i}. Trying sentence fallback...")
                for sent in doc.sents:
                    sent_doc = nlp(sent.text)
                    entities_raw.extend(list(sent_doc.ents))
            
            # Post-process: merge titles (Captain, Dr, etc) into PERSON entities
            entities_list = []
            titles = ["Captain", "Commander", "Dr.", "Doctor", "Professor", "Chief", "Officer", "Lieutenant"]
            for ent in entities_raw:
                start_char = ent.start_char
                # Look back in doc to see if a title precedes this entity
                if start_char > 0:
                    lookback_text = doc.text[:start_char].strip()
                    for title in titles:
                        if lookback_text.endswith(title):
                            # Adjust entity to include title
                            entities_list.append({
                                "text": f"{title} {ent.text}",
                                "label": ent.label_,
                                "start_char": start_char - len(title) - 1,
                                "end_char": ent.end_char
                            })
                            break
                    else:
                        entities_list.append({"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char})
                else:
                    entities_list.append({"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char})

            if entities_list:
                print(f"  [NLP] Detected {len(entities_list)} entities in {chapter_name} chunk {i}")
            
            try:
                resolved_content = doc._.resolved_text if hasattr(doc._, "resolved_text") else doc.text
            except:
                resolved_content = content

            canonical_entities = []
            for ent in entities_list:
                if ent['text'] not in ledger:
                    ledger[ent['text']] = ent['label']
                
                if ent['label'] == 'PERSON':
                    resolved_name = translator.resolve_alias(ent['text'], ledger)
                    if resolved_name:
                        ent['resolved_name'] = resolved_name
                        last_seen_person = resolved_name 
                        ledger[ent['text']] = 'PERSON'
                        ledger[resolved_name] = 'PERSON'
                        if resolved_name not in canonical_entities: canonical_entities.append(resolved_name)
                    else:
                        ent['resolved_name'] = ent['text']
                        ledger[ent['text']] = 'PERSON'
                        if ent['text'] not in canonical_entities: canonical_entities.append(ent['text'])
                else:
                    ent['resolved_name'] = ent['text']
                    if ent['text'] not in canonical_entities: canonical_entities.append(ent['text'])

            all_chunk_triples = []
            if mode in ["spacy", "hybrid"]:
                matcher = Matcher(nlp.vocab)
                matcher.add("TIME", [[{"TEXT": {"REGEX": r"\d{1,2}:\d{2}"}}]])
                current_time = "unknown"
                
                for sent in doc.sents:
                    matches = matcher(sent)
                    for _, start, end in matches: current_time = sent[start:end].text
                    
                    for token in sent:
                        if token.pos_ in ["VERB", "AUX"] or token.dep_ == "ROOT":
                            subj = next((c for c in token.children if c.dep_ in ["nsubj", "nsubjpass", "attr"]), None)
                            subj_text = None
                            
                            if subj:
                                if subj.text.lower() in ["i", "he", "she", "it", "they", "who", "we", "me"]:
                                    subj_text = last_seen_person
                                elif subj.pos_ in ["PROPN", "NOUN"]:
                                    subj_text = subj.text
                            
                            if not subj_text: continue
                            
                            obj_text = ""
                            potential_objs = [c for c in token.children if c.dep_ in ["dobj", "attr", "acomp", "pobj", "prep", "xcomp"]]
                            
                            for obj_node in potential_objs:
                                if obj_node.dep_ == "prep":
                                    pobj = next((gc for gc in obj_node.children if gc.dep_ == "pobj"), None)
                                    if pobj:
                                        obj_text = "".join([t.text_with_ws for t in pobj.subtree]).strip()
                                        break
                                elif obj_node.text == subj_text: 
                                    continue
                                else:
                                    obj_text = "".join([t.text_with_ws for t in obj_node.subtree]).strip()
                                    if obj_text: break
                            
                            if not obj_text: continue

                            label = translator.translate_predicate(token.text, subj_text, obj_text)
                            if label != "NONE":
                                all_chunk_triples.append({
                                    "subject": {"text": subj_text},
                                    "predicate": {"label": label, "text": token.text},
                                    "object": {"text": obj_text},
                                    "timestamp": current_time,
                                    "target_feature": obj_node.lemma_ if label == "HAS_ATTRIBUTE" else "general",
                                    "provenance": {"source_text_snippet": sent.text}
                                })
            
            if mode in ["llm", "hybrid"]:
                llm_triples = extractor.extract_triples_llm(resolved_content, content, canonical_entities=canonical_entities)
                all_chunk_triples.extend(llm_triples)
            
            processed_chunks.append({
                "chapter_name": chapter_name, "chunk_id": i, "content": content,
                "triples": all_chunk_triples, "entities": entities_list
            })
        processed_chapters[chapter_name] = processed_chunks

    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for ent in chunk['entities']:
                base_text = ent.get('resolved_name', ent['text'])
                ent['canonical_id'] = get_canonical_id(base_text)
            for triple in chunk['triples']:
                for role in ['subject', 'object']:
                    node = triple[role]
                    txt = node.get('text', 'UNKNOWN')
                    node['canonical_id'] = get_canonical_id(txt)
    return processed_chapters, ledger

def print_ner_table(processed_chapters):
    data = []
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for ent in chunk['entities']:
                data.append({"Chapter": chapter_name, "Text": ent['text'], "Label": ent.get('label'), "ID": ent.get('canonical_id')})
    if data:
        print(tabulate(pd.DataFrame(data), headers='keys', tablefmt='grid'))
    else:
        print("No entities detected in any chapter.")
