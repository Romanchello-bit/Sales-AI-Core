import random
from graph_module import Graph

def generate_random_graph(num_vertices, density):
    graph = Graph(num_vertices, directed=True)
    for u in range(num_vertices):
        for v in range(num_vertices):
            if u == v:
                continue
            if random.random() < density:
                weight = random.randint(-2, 10)
                graph.add_edge(u, v, weight)
    return graph
