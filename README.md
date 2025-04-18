# Measure Graph Centrality in Social Network

## Overview
This repository contains implementations of graph centrality algorithms for analyzing social networks and identifying influential nodes. We compare different centrality measures (Closeness, Betweenness, Eigenvector, and PageRank) to determine their effectiveness in detecting key nodes within Facebook ego networks.

## Dataset

We utilize the Facebook ego network dataset from the [Stanford Network Analysis Project (SNAP)](https://snap.stanford.edu/data/ego-Facebook.html). This dataset contains:

- Anonymized information about users' ego networks (subgraphs consisting of a user and their direct connections)
- 4,039 nodes and 88,234 edges in the combined network
- Unweighted, undirected edges representing real-world friendship connections

![dataset with ego nodes in orange](./graphs/Facebook%20Dataset.png)

## Algorithms Implemented

### Closeness Centrality

- Quantifies how near a node is to all other nodes in the network
- Identifies nodes that can quickly interact with all others
- Implementation uses Breadth-First Search (BFS) to compute shortest path distances

### Betweenness Centrality

- Implementation of Brandes' algorithm for efficient calculation of betweenness centrality in both directed and undirected graphs.
- Identifies nodes that frequently appear on shortest paths between other nodes.
- Particularly useful for finding nodes that serve as bridges between different communities.

### Eigenvector Centrality

- Implemented using power iteration method.
- Measures node importance based on connections to other important nodes.
- Particularly effective for networks with directed influence patterns.

### PageRank

- Implementation of Google's PageRank algorithm adapted for social networks.
- Models random walks through the network with damping factors (0.85 by default).
- Identifies influential nodes based on the probability of a random surfer visiting them.

## Usage
1. Install all dependencies by running:
    ```bash
    pip install -r requirements.txt
    ```

2. Run the main.py file, on Windows:
    ```bash
    python main.py
    ```
    or on MacOS:
    ```bash
    python3 main.py
    ```

## Key Findings
Our comparative analysis revealed significant differences in how centrality measures identify influential nodes:

1. PageRank performed best, identifying 9 out of 10 ego nodes
2. Betweenness centrality detected 6 ego nodes
3. Closeness centrality found 4 ego nodes
4. Eigenvector centrality only identified 1 ego node

PageRank's effectiveness stems from its balanced consideration of both global and local network properties and its handling of dense community structures.

![PageRank results](./graphs/Pagerank%20Graph.png)

![Betweenness results](./graphs/Betweenness%20Graph.png)

![Closeness results](./graphs/Closeness%20Graph.png)

![Eigenvector results](./graphs/Eigenvector%20Graph.png)