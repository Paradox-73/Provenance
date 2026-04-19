import networkx as nx
from PIL import Image, ImageDraw, ImageFont
import os

def generate_graph_image(nodes, edges, output_path):
    G = nx.Graph()
    for node in nodes:
        G.add_node(node['id'], label=node.get('label', node['id']))
    for edge in edges:
        G.add_edge(edge['from'], edge['to'], label=edge.get('label', ''))
    
    pos = nx.spring_layout(G, k=0.5)
    
    # Scale to image size
    width, height = 1200, 800
    padding = 100
    
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    # Draw edges
    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        px1 = int((x1 + 1) / 2 * (width - 2 * padding) + padding)
        py1 = int((y1 + 1) / 2 * (height - 2 * padding) + padding)
        px2 = int((x2 + 1) / 2 * (width - 2 * padding) + padding)
        py2 = int((y2 + 1) / 2 * (height - 2 * padding) + padding)
        draw.line((px1, py1, px2, py2), fill=(200, 200, 200), width=2)
    
    # Draw nodes
    for node_id, (x, y) in pos.items():
        px = int((x + 1) / 2 * (width - 2 * padding) + padding)
        py = int((y + 1) / 2 * (height - 2 * padding) + padding)
        r = 30
        draw.ellipse((px-r, py-r, px+r, py+r), fill=(100, 150, 250), outline=(50, 50, 50))
        label = G.nodes[node_id]['label']
        draw.text((px-r, py+r+5), label, fill=(0, 0, 0))
        
    img.save(output_path)
    print(f"Saved graph to {output_path}")

if __name__ == "__main__":
    nodes = [{'id': 'A', 'label': 'Kael'}, {'id': 'B', 'label': 'Maria'}]
    edges = [{'from': 'A', 'to': 'B', 'label': 'TALKED_TO'}]
    generate_graph_image(nodes, edges, "test_graph.png")
