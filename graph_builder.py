import pandas as pd
import networkx as nx

print("=" * 70)
print("GRAPH CONSTRUCTION MODULE")
print("=" * 70)

# -------------------------------------------------------
# Load Dataset
# -------------------------------------------------------

data = pd.read_csv("dataset/bcs_data.csv")

print("Dataset Loaded :", data.shape)

# -------------------------------------------------------
# Create Graph
# -------------------------------------------------------

G = nx.Graph()

# -------------------------------------------------------
# Check Required Columns
# -------------------------------------------------------

required_columns = {
    "User": "card1",
    "Device": "DeviceType",
    "IP": "addr1",
    "Merchant": "ProductCD",
    "Transaction": "TransactionID"
}

for name, column in required_columns.items():
    if column not in data.columns:
        print(f"{column} not found. Creating dummy values.")
        data[column] = "Unknown"

# -------------------------------------------------------
# Build Graph
# -------------------------------------------------------

print("\nBuilding Relationship Graph...")

for _, row in data.iterrows():

    user = f"USER_{row['card1']}"
    device = f"DEVICE_{row['DeviceType']}"
    ip = f"IP_{row['addr1']}"
    merchant = f"MERCHANT_{row['ProductCD']}"
    transaction = f"TXN_{row['TransactionID']}"

    # Add Nodes
    G.add_node(user, type="User")
    G.add_node(device, type="Device")
    G.add_node(ip, type="IP")
    G.add_node(merchant, type="Merchant")
    G.add_node(transaction, type="Transaction")

    # Add Relationships
    G.add_edge(user, device)
    G.add_edge(user, ip)
    G.add_edge(user, merchant)
    G.add_edge(user, transaction)

    G.add_edge(device, transaction)
    G.add_edge(ip, transaction)
    G.add_edge(merchant, transaction)

print("\nGraph Construction Completed")

# -------------------------------------------------------
# Graph Statistics
# -------------------------------------------------------

num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

print("Nodes :", num_nodes)
print("Edges :", num_edges)

# -------------------------------------------------------
# Node Statistics
# -------------------------------------------------------

node_types = {
    "User": 0,
    "Device": 0,
    "IP": 0,
    "Merchant": 0,
    "Transaction": 0
}

for node in G.nodes(data=True):

    node_type = node[1]["type"]

    node_types[node_type] += 1

# -------------------------------------------------------
# Save Edge List
# -------------------------------------------------------

edges = pd.DataFrame(G.edges(), columns=["Source", "Target"])

edges.to_csv(
    "dataset/graph_edges.csv",
    index=False
)

# -------------------------------------------------------
# Save Node List
# -------------------------------------------------------

nodes = []

for node, attr in G.nodes(data=True):

    nodes.append({

        "Node": node,

        "Type": attr["type"]

    })

nodes = pd.DataFrame(nodes)

nodes.to_csv(
    "dataset/graph_nodes.csv",
    index=False
)

# -------------------------------------------------------
# Research Table
# -------------------------------------------------------

summary = pd.DataFrame({

    "Graph Component": [

        "Total Nodes",
        "Total Edges",
        "User Nodes",
        "Device Nodes",
        "IP Nodes",
        "Merchant Nodes",
        "Transaction Nodes"

    ],

    "Count": [

        num_nodes,
        num_edges,
        node_types["User"],
        node_types["Device"],
        node_types["IP"],
        node_types["Merchant"],
        node_types["Transaction"]

    ]

})

summary.to_csv(
    "results/tables/graph_statistics.csv",
    index=False
)

print("\nGraph Statistics")

print(summary)

print("\nGraph Builder Completed Successfully")