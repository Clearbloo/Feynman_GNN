from feynman_dataset_builder import FeynmanGraph

graph = FeynmanGraph(3)
graph.add_edge(1, 2)
graph.add_edge(2, 3)
graph.add_edge(3, 1)
graph.connect_global_node()
graph.undirected()

print(graph.edge_index)
graph.build_dfs()
graph.display_graph()