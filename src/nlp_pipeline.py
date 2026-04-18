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

@spacy.Language.component("title_merger")
def title_merger(doc):
    titles = ["Captain", "Commander", "Dr.", "Doctor", "Professor", "Chief", "Officer", "Lieutenant", "Stowaway"]
    with doc.retokenize() as retokenizer:
        for i in range(len(doc) - 1):
            if doc[i].text in titles and doc[i+1].pos_ == "PROPN":
                retokenizer.merge(doc[i:i+2], attrs={"POS": "PROPN", "LABEL": "PERSON"})
    return doc

def load_spacy_model(model_name: str = "en_core_web_trf", ledger: dict = None):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load(model_name)
            nlp.add_pipe("fastcoref")
            nlp.add_pipe("title_merger", before="ner")
            
            # Add custom labels for Environment and Vessel tracking
            ruler = nlp.add_pipe("entity_ruler", before="ner")
            patterns = [
                {"label": "ENVIRONMENT", "pattern": [{"LOWER": "light"}]},
                {"label": "ENVIRONMENT", "pattern": [{"LOWER": "color"}]},
                {"label": "ENVIRONMENT", "pattern": [{"LOWER": "scent"}]},
                {"label": "ENVIRONMENT", "pattern": [{"LOWER": "smell"}]},
                {"label": "VESSEL", "pattern": [{"LOWER": "ship"}]},
                {"label": "VESSEL", "pattern": [{"LOWER": "starlight"}]},
                {"label": "VESSEL", "pattern": [{"LOWER": "void"}, {"LOWER": "voyager"}]},
                {"label": "VESSEL", "pattern": [{"LOWER": "aethelgard"}]}
            ]
            if ledger:
                patterns.extend([{"label": label, "pattern": text} for text, label in ledger.items()])
            ruler.add_patterns(patterns)
        except OSError:
            raise

def extract_triples_deterministic(doc: Doc) -> list[dict]:
    triples = []
    matcher = Matcher(nlp.vocab)
    matcher.add("TIME", [[{"TEXT": {"REGEX": r"\d{1,2}:\d{2}"}}]])
    current_time = "unknown"
    
    for sent in doc.sents:
        matches = matcher(sent)
        for _, start, end in matches:
            current_time = sent[start:end].text
        
        sent_text = sent.text.strip()
        for token in sent:
            # We look for verbs and nouns that anchor relationships
            if token.pos_ in ["VERB", "NOUN", "ADJ"] or token.dep_ == "ROOT" or token.lemma_ in ["be", "become"]:
                # 1. EXTRACT SUBJECT
                subj = None
                for child in token.children:
                    if child.dep_ in ["nsubj", "nsubjpass"]:
                        subj = child
                        break
                
                # If no subject, but the token is an entity (like "Light"), use it as subject
                subj_text = subj.text if subj else ""
                if not subj:
                    for ent in sent.ents:
                        if ent.start <= token.i < ent.end:
                            subj_text = ent.text
                            break
                
                if not subj_text: continue
                
                # Phase 2: Dialogue Coreference Resolution
                if subj_text in ["I", "me", "my"]:
                    parent = token.head if token.pos_ == "VERB" else token
                    if parent.dep_ in ["ccomp", "parataxis"]:
                        speaking_verb = parent.head
                        for child in speaking_verb.children:
                            if child.dep_ in ["nsubj", "nsubjpass"]:
                                subj_text = child.text
                                break
                
                subj_text = subj_text.strip(' .,:;"\'”’!?')

                # Phase 1: Deep Negation Check
                neg_tokens = ["not", "no", "never", "n't"]
                is_negated = any(c.dep_ == "neg" or c.lower_ in neg_tokens for c in token.children) or \
                             any(gc.dep_ == "neg" or gc.lower_ in neg_tokens for c in token.children for gc in c.children)
                label_prefix = "NOT_" if is_negated else ""
                
                # 2. APPLY STRUCTURAL RULES
                
                # A. LOCATED_AT Expansion
                movement_verbs = ["enter", "leave", "visit", "reach", "approach", "exit", "stand"]
                if token.lemma_ in movement_verbs or token.pos_ == "VERB":
                    for child in token.children:
                        if child.dep_ in ["dobj", "pobj", "prep"]:
                            potential_loc = "".join([t.text_with_ws for t in child.subtree]).strip().strip(' .,:;"\'”’!?')
                            # Catch anything that looks like a location or vessel
                            if any(ent.text in potential_loc for ent in sent.ents if ent.label_ in ["LOC", "FAC", "VESSEL", "GPE"]):
                                triples.append({
                                    "subject": {"text": subj_text},
                                    "predicate": {"label": f"{label_prefix}LOCATED_AT", "text": token.text},
                                    "object": {"text": potential_loc},
                                    "timestamp": current_time,
                                    "provenance": {"source_text_snippet": sent_text}
                                })

                # B. POSSESSES / LOST
                if token.lemma_ in ["have", "hold", "possess", "carry", "keep", "lose", "drop"]:
                    rel = "POSSESSES" if token.lemma_ not in ["lose", "drop"] else "LOST"
                    for child in token.children:
                        if child.dep_ == "dobj":
                            obj_text = "".join([t.text_with_ws for t in child.subtree]).strip().strip(' .,:;"\'”’!?')
                            triples.append({
                                "subject": {"text": subj_text},
                                "predicate": {"label": f"{label_prefix}{rel}", "text": token.text},
                                "object": {"text": obj_text},
                                "timestamp": current_time,
                                "provenance": {"source_text_snippet": sent_text}
                            })

                # C. HAS_IDENTITY & HAS_ATTRIBUTE (Role/State Tracking)
                if token.lemma_ in ["be", "become", "remain", "stay", "turn"]:
                    for child in token.children:
                        if child.dep_ in ["attr", "acomp", "prep"]:
                            obj_text = "".join([t.text_with_ws for t in child.subtree]).strip().strip(' .,:;"\'”’!?')
                            
                            # If it looks like a role or name
                            if any(ent.text in obj_text for ent in sent.ents if ent.label_ in ["PERSON", "TITLE"]):
                                triples.append({
                                    "subject": {"text": subj_text},
                                    "predicate": {"label": f"{label_prefix}HAS_IDENTITY", "text": token.text},
                                    "object": {"text": obj_text},
                                    "timestamp": current_time,
                                    "provenance": {"source_text_snippet": sent_text}
                                })
                            # Otherwise it's a state/attribute
                            else:
                                triples.append({
                                    "subject": {"text": subj_text},
                                    "predicate": {"label": f"{label_prefix}HAS_ATTRIBUTE", "text": token.text},
                                    "object": {"text": obj_text},
                                    "target_feature": "state",
                                    "timestamp": current_time,
                                    "provenance": {"source_text_snippet": sent_text}
                                })

                # D. ADJECTIVE MODIFIERS (Direct attributes)
                if token.pos_ == "ADJ" and token.dep_ == "amod" and token.head.pos_ in ["NOUN", "PROPN"]:
                    triples.append({
                        "subject": {"text": token.head.text},
                        "predicate": {"label": "HAS_ATTRIBUTE", "text": "is"},
                        "object": {"text": token.text},
                        "target_feature": "appearance",
                        "timestamp": current_time,
                        "provenance": {"source_text_snippet": sent_text}
                    })
    return triples

def process_chapters_for_nlp(chapter_data: dict, mode="hybrid", ollama_model="qwen2.5:7b", ledger=None) -> tuple:
    if nlp is None: load_spacy_model(ledger=ledger)
    if ledger is None: ledger = {}
    processed_chapters = {}
    
    for chapter_name, chunks in chapter_data.items():
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            content = chunk["content"]
            coref_doc = nlp(content)
            
            # Phase 1: Repair Coreference Resolution
            resolved_content = coref_doc.text
            if coref_doc._.coref_clusters:
                all_mentions = []
                for cluster in coref_doc._.coref_clusters:
                    # Replacement string logic
                    m_start, m_end = cluster[0]
                    main_text = coref_doc.text[m_start:m_end]
                    
                    if len(main_text.split()) > 4 or any(p in main_text.lower() for p in ["that", "who", "which"]):
                        safe_mentions = []
                        for start, end in cluster:
                            text = coref_doc.text[start:end]
                            if not (len(text.split()) > 4 or any(p in text.lower() for p in ["that", "who", "which"])):
                                safe_mentions.append(text)
                        main_text = min(safe_mentions, key=len) if safe_mentions else main_text

                    for start, end in cluster[1:]:
                        all_mentions.append((start, end, main_text))
                
                # Filter overlapping spans
                all_mentions.sort(key=lambda x: (x[0], -(x[1]-x[0])))
                filtered_mentions = []
                last_end = -1
                for start, end, replacement in all_mentions:
                    if start >= last_end:
                        # Possessive handling
                        original_text = coref_doc.text[start:end]
                        if original_text.lower() in ["his", "her", "their", "my", "its"] and not replacement.endswith(("' ", "'s", "’", "’s")):
                            replacement += "'s"
                        filtered_mentions.append((start, end, replacement))
                        last_end = end

                # Sort by start_char descending for replacement
                filtered_mentions.sort(key=lambda x: x[0], reverse=True)
                
                temp_text = coref_doc.text
                for start, end, replacement in filtered_mentions:
                    temp_text = temp_text[:start] + replacement + temp_text[end:]
                resolved_content = temp_text

            doc = nlp(resolved_content)
            entities = [{"text": ent.text, "label": ent.label_, "start_char": ent.start_char, "end_char": ent.end_char} for ent in doc.ents]
            for ent in entities: ledger[ent['text']] = ent['label']

            all_chunk_triples = extract_triples_deterministic(doc)

            processed_chunks.append({
                "chapter_name": chapter_name, "chunk_id": i, "content": content,
                "resolved_content": resolved_content, "entities": entities,
                "triples": all_chunk_triples, "coref_clusters": coref_doc._.coref_clusters
            })
        processed_chapters[chapter_name] = processed_chunks
    
    known_ids = list(ledger.keys())
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for ent in chunk['entities']:
                matches = difflib.get_close_matches(ent['text'], known_ids, n=1, cutoff=0.8)
                res = matches[0] if matches else ent['text']
                cid = re.sub(r'[^A-Z0-9_]', '', res.upper().replace(' ', '_').replace("'", ""))
                ent['canonical_id'] = cid.split('_')[-1] if ent.get('label') in ['PERSON', 'TITLE'] else cid
            for triple in chunk['triples']:
                for role in ['subject', 'object']:
                    node = triple[role]
                    txt = node.get('text', 'UNKNOWN')
                    matches = difflib.get_close_matches(txt, known_ids, n=1, cutoff=0.8)
                    res = matches[0] if matches else txt
                    cid = re.sub(r'[^A-Z0-9_]', '', res.upper().replace(' ', '_').replace("'", ""))
                    triple[role]['canonical_id'] = cid
    return processed_chapters, ledger

def print_ner_table(processed_chapters):
    data = []
    for chapter_name, chunks in processed_chapters.items():
        for chunk in chunks:
            for ent in chunk['entities']:
                data.append({"Chapter": chapter_name, "Text": ent['text'], "Label": ent.get('label'), "ID": ent.get('canonical_id')})
    print(tabulate(pd.DataFrame(data), headers='keys', tablefmt='grid'))
