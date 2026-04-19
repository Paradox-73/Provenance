import json
import os

def main():
    output_file = "ground_truth_old.json"
    
    ground_truth = {
      "stories": [
        {
          "story_id": "Chapter_1",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Temporal/Spatial",
              "entity_involved": "Kael",
              "source_sentence_1": "Commander Kael stood directly beside her at the tactical station, tapping commands into the console.",
              "source_sentence_2": "\"Security Log: Commander Kael entered the Engineering Deck at 08:00 hours.\""
            },
            {
              "conflict_type": "Attribute",
              "entity_involved": "Kael's Scar",
              "source_sentence_1": "The jagged scar running down his left cheek seemed to glow in the red emergency lighting of the Bridge.",
              "source_sentence_2": "His skin was perfect, revealing a cheek that had never known a wound or a scar."
            },
            {
              "conflict_type": "Inventory",
              "entity_involved": "Red Onyx Key",
              "source_sentence_1": "He reached into his belt pouch and pulled out the Red Onyx Key.",
              "source_sentence_2": "\"I seem to have lost the Red Onyx Key. I haven't seen it since yesterday.\""
            },
            {
              "conflict_type": "Identity",
              "entity_involved": "Doctor",
              "source_sentence_1": "Dr. Aris was waiting for them by the stasis pods, looking weary.",
              "source_sentence_2": "\"Good work, Dr. Evans. Keep me posted on any changes.\""
            }
          ]
        },
        {
          "story_id": "Chapter_2",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Temporal/Spatial",
              "entity_involved": "Maria",
              "source_sentence_1": "At 10:00 hours, Maria entered the Captain’s Quarters, the heavy doors hissing shut behind her.",
              "source_sentence_2": "Maria was pacing the Hangar Bay, the cold draft of the vacuum-seal doors biting at her skin."
            },
            {
              "conflict_type": "Attribute",
              "entity_involved": "Water",
              "source_sentence_1": "She sat at her desk, her hands trembling as she reached for a glass of water.",
              "source_sentence_2": "she was still holding the glass of water from her desk, but the liquid was now frozen solid."
            }
          ]
        },
        {
          "story_id": "Chapter_3",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Identity",
              "entity_involved": "Aris",
              "source_sentence_1": "Aris, you’re the ship’s Medical Doctor.",
              "source_sentence_2": "Aris is the ship's Chief Engineer, after all."
            },
            {
              "conflict_type": "Attribute",
              "entity_involved": "Aris/Kael Scar",
              "source_sentence_1": "The jagged scar running down his left cheek seemed to glow in the red emergency lighting of the Bridge.",
              "source_sentence_2": "Maria saw the jagged scar on his left cheek—the same scar that had belonged to Kael."
            }
          ]
        },
        {
          "story_id": "Chapter_4",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Attribute",
              "entity_involved": "Red Onyx Key",
              "source_sentence_1": "The Red Onyx Key was a small, smooth stone, no larger than a child’s marble.",
              "source_sentence_2": "In its place stood a large, jagged crystalline shard, its edges so sharp they seemed to cut the very air around it."
            }
          ]
        },
        {
          "story_id": "Chapter_5",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Inventory",
              "entity_involved": "Red Onyx Key",
              "source_sentence_1": "He clearly possessed the artifact, yet he made no move to give it to her.",
              "source_sentence_2": "\"But I also lost the Red Onyx Key in the reactor core earlier this morning.\""
            }
          ]
        },
        {
          "story_id": "Chapter_6",
          "genre": "Sci-Fi",
          "inconsistencies": [
            {
              "conflict_type": "Temporal/Spatial",
              "entity_involved": "Maria",
              "source_sentence_1": "At 12:00 hours, she was standing on the Bridge, her hand resting on the tactical console that was both smooth and jagged, small and large.",
              "source_sentence_2": "But at the same time, at that very same 12:00 hours, Maria was in the Engine Room, the roar of the reactor core deafening in her ears."
            },
            {
              "conflict_type": "Attribute",
              "entity_involved": "Kael's Eyes",
              "source_sentence_1": "At 12:00, Kael’s eyes were a deep, piercing blue, the color of a summer sky on a world she had long since forgotten.",
              "source_sentence_2": "They were no longer blue; they were a bright, unnatural green, the color of the nebula outside the viewscreen."
            }
          ]
        }
      ]
    }
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(ground_truth, f, indent=2)
    
    print(f"Successfully created {output_file}")

if __name__ == "__main__":
    main()
