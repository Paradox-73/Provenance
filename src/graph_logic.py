from .neo4j_adapter import Neo4jAdapter
from .nlp_pipeline import run_nli_check
from sentence_transformers import SentenceTransformer, util
import requests
import json
import networkx as nx
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import os

# IDENTITY-AWARE QUERIES (Uses SAME_AS links)
IMPOSSIBLE_LOCATION_QUERY = """
MATCH (p:Entity)-[:SAME_AS*0..1]-(e1:Entity)
MATCH (p)-[:SAME_AS*0..1]-(e2:Entity)
MATCH (e1)-[r1:LOCATED_AT]->(l1)
MATCH (e2)-[r2:LOCATED_AT]->(l2)
WHERE id(l1) < id(l2) 
AND (
    (r1.chapter = r2.chapter AND (r1.timestamp = r2.timestamp OR r1.timestamp = 'unknown' OR r2.timestamp = 'unknown'))
    OR 
    (r1.timestamp <> 'unknown' AND r1.timestamp = r2.timestamp)
)
RETURN p.name as character_name, l1.name as location1, l2.name as location2, r1.chapter as conflict_time, r1.text as source1, r2.text as source2
"""

INVENTORY_CONFLICT_QUERY = """
MATCH (p:Entity)-[:SAME_AS*0..1]-(e1:Entity)
MATCH (p)-[:SAME_AS*0..1]-(e2:Entity)
MATCH (e1)-[:HAS_ASSERTION]->(a1:Assertion)-[:ABOUT]->(item:Entity)
MATCH (e2)-[:HAS_ASSERTION]->(a2:Assertion)-[:ABOUT]->(item)
WHERE id(a1) < id(a2)
AND ((a1.label = 'POSSESSES' AND a2.label = 'LOST') OR (a1.label = 'LOST' AND a2.label = 'POSSESSES'))
RETURN p.name as character_name, item.name as item_name, a1.text as source1, a2.text as source2, a1.chapter as conflict_time
"""

ATTRIBUTE_CONFLICT_QUERY = """
MATCH (p:Entity)-[:SAME_AS*0..1]-(e1:Entity)
MATCH (p)-[:SAME_AS*0..1]-(e2:Entity)
MATCH (e1)-[r1:HAS_ATTRIBUTE]->(attr1:Entity)
MATCH (e2)-[r2:HAS_ATTRIBUTE]->(attr2:Entity)
WHERE id(attr1) < id(attr2) AND r1.target_feature = r2.target_feature
RETURN p.name as character_name, attr1.name as attribute1, attr2.name as attribute2, r1.target_feature as target_feature, r1.text as source1, r2.text as source2
"""

IDENTITY_CONFLICT_QUERY = """
MATCH (p:Entity)-[:SAME_AS*0..1]-(e1:Entity)
MATCH (p)-[:SAME_AS*0..1]-(e2:Entity)
MATCH (e1)-[r1:HAS_IDENTITY]->(id1:Entity)
MATCH (e2)-[r2:HAS_IDENTITY]->(id2:Entity)
WHERE id(id1) < id(id2)
RETURN p.name as character_name, id1.name as identity1, id2.name as identity2, r1.text as source1, r2.text as source2
"""

class ConflictDetector:
    def __init__(self, neo4j_adapter: Neo4jAdapter):
        self.neo4j_adapter = neo4j_adapter
        self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _format_inconsistency(self, conflict_type: str, details: dict, confidence: float) -> dict:
        return {"type": conflict_type, "description": f"Detected {conflict_type}.", "details": details, "confidence": confidence}

    def audit_temporal_logic(self) -> list[dict]:
        print("Auditing temporal logic (Agentic)...")
        query = "MATCH (p:Entity)-[r:LOCATED_AT]->(l:Entity) WHERE r.timestamp <> 'unknown' RETURN p.name as char, r.timestamp as time, l.name as loc, r.text as source, r.chapter as chap ORDER BY r.chapter, r.timestamp"
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        char_history = {}
        for rec in results:
            char = rec['char']
            if char not in char_history: char_history[char] = []
            char_history[char].append(rec)
        for history in char_history.values():
            for i in range(len(history) - 1):
                cur, nxt = history[i], history[i+1]
                if cur['chap'] == nxt['chap'] and cur['time'] > nxt['time']:
                    inconsistencies.append(self._format_inconsistency("Temporal Paradox", {"char": cur['char'], "before": cur['time'], "after": nxt['time'], "sources": [cur['source'], nxt['source']]}, 1.0))
        return inconsistencies

    def audit_inventory_logic(self) -> list[dict]:
        print("Auditing spatial/volume logic...")
        query = "MATCH (p:Entity)-[:HAS_ASSERTION]->(a:Assertion)-[:ABOUT]->(item:Entity) WHERE a.label = 'POSSESSES' RETURN p.name as char, item.name as item, a.text as source"
        results = self.neo4j_adapter.run_cypher_query(query)
        inconsistencies = []
        
        for rec in results:
            item = rec['item']
            context = rec['source']
            
            prompt = f"""Analyze the physical possibility:
Relationship: {rec['char']} possesses {item}.
Context from text: "{context}"

Is it physically impossible for this item to be in the mentioned container or carried this way (e.g., a car in a pocket)? 
Only flag extreme spatial impossibilities.
Return ONLY JSON: {{"is_impossible": true/false, "reason": "brief explanation"}}
"""
            payload = {"model": "qwen2.5:7b", "prompt": prompt, "stream": False, "format": "json"}
            try:
                r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
                res = json.loads(r.json()['response'])
                if res.get("is_impossible"):
                    inconsistencies.append(self._format_inconsistency(
                        "Spatial Impossibility", 
                        {"char": rec['char'], "item": item, "reason": res.get("reason"), "context": context}, 
                        0.9
                    ))
            except:
                pass
        return inconsistencies

    def detect_temporal_inconsistency(self) -> list[dict]:
        print("Detecting overlapping location conflicts...")
        results = self.neo4j_adapter.run_cypher_query(IMPOSSIBLE_LOCATION_QUERY)
        inconsistencies = []
        for rec in results:
            nli = run_nli_check(f"{rec['character_name']} was at {rec['location1']}.", f"{rec['character_name']} was at {rec['location2']}.")
            if nli["prediction"] == "contradiction":
                inconsistencies.append(self._format_inconsistency("Impossible Location", {"char": rec['character_name'], "locs": [rec['location1'], rec['location2']], "sources": [rec['source1'], rec['source2']]}, nli["confidence"]))
        return inconsistencies

    def check_attribute_compatibility(self, attr1, attr2, feature):
        prompt = f"Feature: {feature}. Described as '{attr1}' and '{attr2}'. Are these contradictory (e.g. red/blue)? Or compatible (e.g. jagged/crystalline)? Return ONLY JSON: {{\"is_contradictory\": true/false}}"
        payload = {"model": "qwen2.5:7b", "prompt": prompt, "stream": False, "format": "json"}
        try:
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=5)
            return json.loads(r.json()['response']).get("is_contradictory", True)
        except: return True

    def detect_attribute_inconsistency(self) -> list[dict]:
        print("Detecting attribute inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(ATTRIBUTE_CONFLICT_QUERY)
        inconsistencies = []
        for rec in results:
            if self.check_attribute_compatibility(rec['attribute1'], rec['attribute2'], rec['target_feature']):
                nli = run_nli_check(f"The {rec['target_feature']} is {rec['attribute1']}.", f"The {rec['target_feature']} is {rec['attribute2']}.")
                if nli["prediction"] == "contradiction":
                    inconsistencies.append(self._format_inconsistency("Attribute Conflict", {"char": rec['character_name'], "feat": rec['target_feature'], "vals": [rec['attribute1'], rec['attribute2']], "sources": [rec['source1'], rec['source2']]}, nli["confidence"]))
        return inconsistencies

    def detect_inventory_inconsistency(self) -> list[dict]:
        print("Detecting inventory inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(INVENTORY_CONFLICT_QUERY)
        return [self._format_inconsistency("Inventory Conflict", {"char": r['character_name'], "item": r['item_name'], "sources": [r['source1'], r['source2']]}, 1.0) for r in results]

    def detect_identity_inconsistency(self) -> list[dict]:
        print("Detecting identity inconsistencies...")
        results = self.neo4j_adapter.run_cypher_query(IDENTITY_CONFLICT_QUERY)
        inconsistencies = []
        
        # JUNK FILTER: Skip identities that are clearly just phrases or junk
        junk_words = ["what", "you", "him", "her", "me", "it", "this", "that", "something", "anything", "nothing"]
        
        for rec in results:
            id1, id2 = rec['identity1'].lower(), rec['identity2'].lower()
            if any(j == id1 or j == id2 for j in junk_words) or len(id1) < 3 or len(id2) < 3:
                continue

            # Check if they are actually different roles (e.g. Doctor vs Engineer)
            # Use NLI to see if one identity contradicts being the other
            nli = run_nli_check(f"{rec['character_name']} is a {rec['identity1']}.", f"{rec['character_name']} is a {rec['identity2']}.")
            if nli["prediction"] == "contradiction" and nli["confidence"] > 0.8:
                inconsistencies.append(self._format_inconsistency("Identity Conflict", {"char": rec['character_name'], "ids": [rec['identity1'], rec['identity2']], "sources": [rec['source1'], rec['source2']]}, nli["confidence"]))
        return inconsistencies

    def detect_all_inconsistencies(self) -> list[dict]:
        all_c = []
        all_conflicts = [
            self.audit_temporal_logic, 
            self.audit_inventory_logic, 
            self.detect_temporal_inconsistency, 
            self.detect_inventory_inconsistency, 
            self.detect_attribute_inconsistency,
            self.detect_identity_inconsistency
        ]
        for func in all_conflicts:
            try: all_c.extend(func())
            except Exception as e: print(f"Error in {func.__name__}: {e}")
        return all_c

    def visualize_graph(self, output_path: str = "visualizations/graph_viz.png", title: str = "Narrative Knowledge Graph"):
        print(f"Saving graph snapshot to {output_path}...")
        
        # Ensure directory exists
        dir_name = os.path.dirname(output_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)

        query = """
        MATCH (n)-[r]->(m) 
        RETURN COALESCE(n.name, n.canonical_id, 'Unknown') as start, 
               labels(n) as s_labels,
               type(r) as rel, 
               COALESCE(m.name, m.canonical_id, 'Unknown') as end,
               labels(m) as e_labels
        LIMIT 100
        """
        results = self.neo4j_adapter.run_cypher_query(query)
        if not results:
            print("No data found for visualization.")
            return

        G = nx.MultiDiGraph()
        node_colors = {}

        def get_color(labels):
            if 'Person' in labels: return '#ff9999' # Red
            if 'Location' in labels: return '#99ff99' # Green
            return '#9999ff' # Blue (Item/Entity)

        for record in results:
            s, e = str(record['start']), str(record['end'])
            # CLEANING: Truncate long labels and skip sentences
            if len(s) > 25 or len(e) > 25 or s == "Unknown" or e == "Unknown": continue
            
            G.add_edge(s, e, label=record['rel'])
            node_colors[s] = get_color(record['s_labels'])
            node_colors[e] = get_color(record['e_labels'])
        
        if len(G.nodes) == 0:
            print("Graph has no valid nodes to display.")
            return

        plt.figure(figsize=(20, 12))
        pos = nx.spring_layout(G, k=1.5, iterations=50) # Spread out more
        
        colors = [node_colors.get(node, '#cccccc') for node in G.nodes()]
        
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=3000, 
                font_size=10, font_weight='bold', edge_color='#bbbbbb', 
                arrows=True, arrowsize=20, connectionstyle='arc3, rad = 0.1')
        
        edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, label_pos=0.3)
        
        plt.title(title, fontsize=15)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Graph visualization saved to {output_path}")
