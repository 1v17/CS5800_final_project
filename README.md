# Measure Graph Centrality in Social Network

## Overview
This repository contains implementations of various graph centrality algorithms for detecting influential nodes in social networks. The project focuses on comparing Betweenness Centrality, Eigenvector Centrality, and PageRank algorithms to determine their effectiveness in identifying key nodes within network structures.

## Algorithms Implemented

### Closeness Centrality

<!-- TODO: Add descriptions for closeness centrality. -->

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
- Models random walks through the network with damping factors.
- Identifies influential nodes based on the probability of a random surfer visiting them.

## Dataset Support

<!-- TODO: Add descriptions of dataset. -->

## Usage
Installation
```bash
pip install -r requirements.txt
```

<!-- TODO: generate requirements.txt after finishing implementation. -->

## Results and Discussions

<!-- TODO: add results here. -->