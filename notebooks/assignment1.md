### Code for Assignment 1 of Course 02467 at DTU
#### Computational Social Scientists Network Analysis

```python
import pandas as pd
import re
import networkx as nx
from networkx.readwrite import json_graph
from itertools import combinations
from collections import defaultdict
import json
import community as community_louvain

def clean_author_id(author_id):
    if pd.isna(author_id) or not str(author_id).strip():
        return None
    match = re.search(r'(?:https://openalex.org/)?([A-Z]\d+)$', str(author_id).strip())
    return match.group(1) if match else None

def process_papers(papers_df):
    collaborations = defaultdict(int)

    for _, row in papers_df.iterrows():
        try:
            author_ids = eval(row['author_ids']) if isinstance(row['author_ids'], str) else row['author_ids']
        except:
            author_ids = []
            
        cleaned_authors = [clean_author_id(aid) for aid in author_ids]
        valid_authors = [aid for aid in cleaned_authors if aid is not None]
        
        for author1, author2 in combinations(sorted(valid_authors), 2):
            collaborations[(author1, author2)] += 1
            
    return [(a1, a2, w) for (a1, a2), w in collaborations.items()]

def add_author_attributes(G, authors_df):
    for _, author_data in authors_df.iterrows():
        author_id = clean_author_id(author_data['OpenAlex ID'])
        if author_id and author_id in G:
            attributes = {
                'display_name': author_data.get('Display Name', ''),
                'h_index': author_data.get('H-Index', 0),
                'works_count': author_data.get('Works Count', 0),
                'country_code': author_data.get('Country Code', ''),
                'citation_count': author_data.get('Cited By Count', 0),
                'first_publication_year': author_data.get('First Publication Year', None)
            }
            nx.set_node_attributes(G, {author_id: attributes})

def detect_communities(G):
    partition = community_louvain.best_partition(G)
    nx.set_node_attributes(G, partition, 'community')
    
    num_communities = max(partition.values()) + 1
    community_sizes = pd.Series(partition.values()).value_counts().sort_index()
    
    print(f"\nCommunity detection results:")
    print(f"Number of communities detected: {num_communities}")
    print("Community sizes:")
    print(community_sizes)
    
    return partition

def main():
    papers_df = pd.read_csv("papers.csv")
    authors_df = pd.read_csv("authors.csv")
    
    weighted_edgelist = process_papers(papers_df)
    
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edgelist)
    add_author_attributes(G, authors_df)
    partition = detect_communities(G)
    community_data = [{
        'author_id': node,
        'community': partition[node],
        'display_name': G.nodes[node].get('display_name', ''),
        'country_code': G.nodes[node].get('country_code', '')
    } for node in G.nodes()]
    
    pd.DataFrame(community_data).to_csv('community_assignments.csv', index=False)
    
    # Save to JSON (now includes community information)
    graph_data = json_graph.node_link_data(G)
    with open('collaboration_network.json', 'w') as json_file:
        json.dump(graph_data, json_file, indent=4)
    
    # save edgelist as CSV
    edgelist_df = pd.DataFrame(weighted_edgelist, columns=['author1', 'author2', 'weight'])
    edgelist_df.to_csv('weighted_edgelist.csv', index=False)
    
    print(f"\nNetwork created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

if __name__ == "__main__":
    main()
