import pandas as pd
import networkx as nx
import numpy as np

print("="*70)
print("GRAPH TRUST SCORE (GTS)")
print("="*70)

# -------------------------------------------------------
# Load Graph
# -------------------------------------------------------

nodes = pd.read_csv("dataset/graph_nodes.csv")
edges = pd.read_csv("dataset/graph_edges.csv")

print("Nodes :", len(nodes))
print("Edges :", len(edges))

# -------------------------------------------------------
# Build Graph
# -------------------------------------------------------

G = nx.Graph()

for _, row in nodes.iterrows():

    G.add_node(
        row["Node"],
        node_type=row["Type"]
    )

for _, row in edges.iterrows():

    G.add_edge(
        row["Source"],
        row["Target"]
    )

print("Graph Loaded Successfully")

# -------------------------------------------------------
# Initial Trust Assignment
# -------------------------------------------------------

trust = {}

for node in G.nodes():

    if node.startswith("USER"):

        trust[node] = 0.90

    elif node.startswith("DEVICE"):

        trust[node] = 0.80

    elif node.startswith("IP"):

        trust[node] = 0.75

    elif node.startswith("MERCHANT"):

        trust[node] = 0.85

    else:

        trust[node] = 0.70

# -------------------------------------------------------
# Connectivity Trust Score
# -------------------------------------------------------

degree = dict(G.degree())

max_degree = max(degree.values())

connectivity = {}

for node in G.nodes():

    connectivity[node] = degree[node] / max_degree

# -------------------------------------------------------
# Neighbor Trust Score
# -------------------------------------------------------

neighbor_score = {}

for node in G.nodes():

    neighbors = list(G.neighbors(node))

    if len(neighbors) == 0:

        neighbor_score[node] = trust[node]

    else:

        scores = []

        for n in neighbors:

            scores.append(trust[n])

        neighbor_score[node] = np.mean(scores)

# -------------------------------------------------------
# Graph Trust Score
# -------------------------------------------------------

graph_trust = {}

for node in G.nodes():

    graph_trust[node] = (

        0.40 * trust[node]

        +

        0.30 * connectivity[node]

        +

        0.30 * neighbor_score[node]

    )

# -------------------------------------------------------
# Save Scores
# -------------------------------------------------------

graph_df = pd.DataFrame({

    "Node": list(graph_trust.keys()),

    "GraphTrustScore": list(graph_trust.values()),

    "InitialTrust": [trust[i] for i in graph_trust.keys()],

    "ConnectivityScore": [connectivity[i] for i in graph_trust.keys()],

    "NeighborTrust": [neighbor_score[i] for i in graph_trust.keys()]

})

graph_df.to_csv(

    "dataset/graph_trust_scores.csv",

    index=False

)

# -------------------------------------------------------
# Statistics
# -------------------------------------------------------

summary = pd.DataFrame({

    "Metric":[

        "Average Trust",

        "Maximum Trust",

        "Minimum Trust",

        "Trusted Nodes (>0.80)",

        "Medium Trust (0.60-0.80)",

        "Low Trust (<0.60)"

    ],

    "Value":[

        round(graph_df["GraphTrustScore"].mean(),4),

        round(graph_df["GraphTrustScore"].max(),4),

        round(graph_df["GraphTrustScore"].min(),4),

        len(graph_df[graph_df["GraphTrustScore"]>0.80]),

        len(graph_df[
            (graph_df["GraphTrustScore"]>=0.60) &
            (graph_df["GraphTrustScore"]<=0.80)
        ]),

        len(graph_df[graph_df["GraphTrustScore"]<0.60])

    ]

})

summary.to_csv(

    "results/tables/graph_trust_results.csv",

    index=False

)

print(summary)

print("\nGraph Trust Score Generated Successfully")