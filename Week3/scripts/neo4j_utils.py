"""Shared Neo4j connection and GDS utilities for Week3 scripts."""

from neo4j import GraphDatabase
import os

URI = "bolt://localhost:7687"
USER = "neo4j"
PASSWORD = "s3cureP@ssword"

def get_driver():
    return GraphDatabase.driver(URI, auth=(USER, PASSWORD))

def run_query(query, params=None):
    driver = get_driver()
    with driver.session(database="neo4j") as session:
        result = list(session.run(query, params or {}))
    driver.close()
    return result

def get_gds():
    from graphdatascience import GraphDataScience
    return GraphDataScience(URI, auth=(USER, PASSWORD), database="neo4j")

def output_path(*parts):
    """Return absolute path under outputs/ folder."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "outputs", *parts)
