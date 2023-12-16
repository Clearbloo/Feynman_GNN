from physics.Feynman_GNN.feynman_code.src.base_feynman_graph import FeynmanGraph

graph = FeynmanGraph(edges=[(1,2),(2,3),(2,4),(4,5),(4,6)])

# graph.connect_global_node()
graph.make_edges_undirected()

for i in range(1,7):
    graph.add_node_feat(str(i), [1,0,0])

print(graph.edge_index)
graph.build_dfs()
graph.display_graph()