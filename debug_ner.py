import spacy
from spacy.tokens import Doc
from fastcoref import spacy_component

def test_ner():
    model_name = "en_core_web_trf"
    print(f"Loading {model_name}...")
    nlp = spacy.load(model_name)
    
    text = "Captain Maria checked the digital chronometer on the wall; it read 08:00 hours. She stood on the Bridge of the Starship Aethelgard."
    
    print("\n--- Test 1: Standard NER ---")
    doc = nlp(text)
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

    print("\n--- Test 2: With title_merger ---")
    @spacy.Language.component("title_merger_test")
    def title_merger(doc):
        titles = ["Captain", "Commander", "Dr.", "Doctor", "Professor", "Chief", "Officer", "Lieutenant"]
        with doc.retokenize() as retokenizer:
            for i in range(len(doc) - 1):
                if doc[i].text in titles and doc[i+1].pos_ == "PROPN":
                    retokenizer.merge(doc[i:i+2], attrs={"POS": "PROPN", "ENT_TYPE": "PERSON"})
        return doc
    
    nlp.add_pipe("title_merger_test", before="ner")
    doc = nlp(text)
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")

    print("\n--- Test 3: With fastcoref ---")
    nlp.add_pipe("fastcoref")
    doc = nlp(text)
    print(f"Entities: {[(ent.text, ent.label_) for ent in doc.ents]}")
    if hasattr(doc._, "resolved_text"):
        print(f"Resolved: {doc._.resolved_text}")

if __name__ == "__main__":
    test_ner()
