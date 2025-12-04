class Graph:
    def __init__(self, num_vertices, directed=True):
        self.num_vertices = num_vertices
        self.directed = directed
        self.adj_matrix = [[None] * num_vertices for _ in range(num_vertices)]
        self.adj_list = {i: [] for i in range(num_vertices)}

    def add_edge(self, u, v, weight):
        if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
            self.adj_list[u].append((v, weight))
            self.adj_matrix[u][v] = weight
            if not self.directed:
                self.adj_list[v].append((u, weight))
                self.adj_matrix[v][u] = weight
        else:
            raise ValueError(f"Vertex index out of bounds: {u}, {v}")

    def get_matrix(self):
        return self.adj_matrix

    def get_list(self):
        return self.adj_list
