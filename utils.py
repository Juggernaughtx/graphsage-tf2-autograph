import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph

version_info = list(map(int, nx.__version__.split(".")))
major = version_info[0]
minor = version_info[1]
assert (major <= 1) and (minor <= 11), "networkx major version > 1.11"


def load_data(prefix, normalize=True, load_walks=False):
    print("R50")
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)

    feats = np.load(prefix + "-feats.npy")
    
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {k: int(v) for k, v in id_map.items()}

    class_map = json.load(open(prefix + "-class_map.json"))
    class_map = {id_map[k]: v for k, v in class_map.items()}
    class_map = [v for k, v in sorted(class_map.items())]
    class_map = np.asarray(class_map, dtype=np.int32)

    adj, deg = construct_adj(G, id_map)

    test_nodes = [id_map[n] for n in G.nodes() if G.node[n]["test"]]
    val_nodes = [id_map[n]  for n in G.nodes() if G.node[n]["val"]]
    train_nodes = [id_map[n]  for n in G.nodes() if not G.node[n]["val"] and not G.node[n]["test"]]
    train_nodes = [n for n in train_nodes if deg[n] > 0]

    print(f"train_nodes: {len(train_nodes)}")
    print(f"val_nodes: {len(val_nodes)}")
    print(f"test_nodes: {len(test_nodes)}")

    return feats, train_nodes, val_nodes, test_nodes, class_map, adj


def construct_adj(G, id2idx):
    adj = len(id2idx) * np.ones((len(id2idx) + 1, 25), dtype=np.int32)
    deg = np.zeros((len(id2idx),))

    for nodeid in G.nodes():
        neighbors = np.array(
            [
                id2idx[neighbor]
                for neighbor in G.neighbors(nodeid)
            ]
        )
        deg[id2idx[nodeid]] = len(neighbors)
        if len(neighbors) == 0:
            continue
        if len(neighbors) > 25:
            neighbors = np.random.choice(neighbors, 25, replace=False)
        elif len(neighbors) < 25:
            neighbors = np.random.choice(neighbors, 25, replace=True)
        adj[id2idx[nodeid], :] = neighbors
    return adj, deg