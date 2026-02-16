# src/friction_queries.py

IMPOSSIBLE_LOCATION_QUERY = """
MATCH (p:Person)-[r1:LOCATED_AT]->(l1:Location)
MATCH (p)-[r2:LOCATED_AT]->(l2:Location)
WHERE
    r1.timestamp = r2.timestamp AND l1.id <> l2.id
RETURN
    p.name AS character_name,
    r1.timestamp AS conflict_time,
    l1.name AS location1,
    l2.name AS location2,
    r1.source_text AS source1,
    r2.source_text AS source2
"""
