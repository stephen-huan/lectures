import networkx as nx
from network2tikz import plot

#   [1, 2, 3, 4, 5]
l = list(map(str, [3, 5, 1, 2, 4]))

nodes = l
edges = []
for i in range(len(l)):
    edges.append((l[i], str(i + 1)))
# edges = [('a','b'), ('a','c'), ('c','d'),('d','b')]

G = nx.DiGraph()
G.add_nodes_from(nodes)
G.add_edges_from(edges)

style = {}
style['node_label'] = l
# style['node_color'] = [colors[g] for g in gender]
# style['node_opacity'] = .5
style['edge_curved'] = .1
style["edge_directed"] = True

plot((nodes, edges),'network.tex',**style, canvas=(5, 5), layout="random", force=3/4)
