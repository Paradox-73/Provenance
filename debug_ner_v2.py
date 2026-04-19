import spacy
from spacy.tokens import Doc, Span
from fastcoref import spacy_component

def merge_titles_in_ents(doc):
    """Manually merges titles like 'Captain' into adjacent PERSON entities."""
    titles = ["Captain", "Commander", "Dr.", "Doctor", "Professor", "Chief", "Officer", "Lieutenant"]
    new_ents = []
    ents = list(doc.ents)
    i = 0
    while i < len(ents):
        ent = ents[i]
        start_token_idx = ent.start
        if start_token_idx > 0:
            prev_token = doc[start_token_idx - 1]
            if prev_token.text in titles:
                new_ent = Span(doc, start_token_idx - 1, ent.end, label=ent.label)
                new_ents.append(new_ent)
                i += 1
                continue
        new_ents.append(ent)
        i += 1
    try:
        doc.ents = spacy.util.filter_spans(new_ents)
    except:
        pass
    return doc

def test_ner():
    model_name = "en_core_web_trf"
    print(f"Loading {model_name}...")
    nlp = spacy.load(model_name)
    
    text = "Captain Maria checked the digital chronometer on the wall; it read 08:00 hours. She stood on the Bridge of the Starship Aethelgard."
    
    print("\n--- Test 1: Standard NER ---")
    doc = nlp(text)
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

    print("\n--- Test 2: With manual merge_titles_in_ents ---")
    doc = nlp(text)
    doc = merge_titles_in_ents(doc)
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

if __name__ == "__main__":
    test_ner()
