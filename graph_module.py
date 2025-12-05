class Graph:
    def __init__(self, num_vertices, directed=True):
        self.num_vertices = num_vertices
        self.directed = directed
        # Using float('inf') for no connection as requested
        self.adj_matrix = [[float('inf')] * num_vertices for _ in range(num_vertices)]
        self.adj_list = {i: [] for i in range(num_vertices)}

    def add_edge(self, u, v, weight):
        if 0 <= u < self.num_vertices and 0 <= v < self.num_vertices:
            # Directed edge logic
            self.adj_list[u].append((v, weight))
            self.adj_matrix[u][v] = weight
            
            if not self.directed:
                self.adj_list[v].append((u, weight))
                self.adj_matrix[v][u] = weight
        else:
            raise ValueError(f"Vertex index out of bounds: {u}, {v}")

    def to_adjacency_matrix(self):
        """
        Returns a 2D list (matrix) of size V x V.
        matrix[u][v] = weight if edge exists.
        matrix[u][v] = float('inf') if no edge exists.
        matrix[u][u] = 0 (distance to self).
        """
        # Create a deep copy to avoid modifying internal state if needed, 
        # or just return internal state if we maintain it strictly.
        # But user requested "matrix[u][u] = 0". Our internal init uses float('inf').
        # So we should probably update internal or return a modified copy.
        # Let's return a generated one if we want to be safe, or update internal.
        # Updating internal is better for consistency if we use it.
        # But wait, internal initialized with inf. 
        # Let's just fix diagonals in internal if they are inf.
        
        for i in range(self.num_vertices):
            self.adj_matrix[i][i] = 0
            
        return self.adj_matrix

    def from_adjacency_matrix(self, matrix):
        """
        Clears the current graph.
        Populates the adjacency list based on the matrix values.
        """
        self.num_vertices = len(matrix)
        self.adj_matrix = matrix
        self.adj_list = {i: [] for i in range(self.num_vertices)}
        
        for i in range(self.num_vertices):
            for j in range(self.num_vertices):
                w = matrix[i][j]
                if i != j and w != float('inf'):
                    self.adj_list[i].append((j, w))

    def get_matrix(self):
        return self.adj_matrix

    def get_list(self):
        return self.adj_list

    def check_connectivity(self, start_node_id):
        """
        Checks if all nodes are reachable from start_node_id using BFS.
        Returns: (is_connected: bool, unreachable_nodes: list)
        """
        if start_node_id < 0 or start_node_id >= self.num_vertices:
            return False, []

        visited = set()
        queue = [start_node_id]
        visited.add(start_node_id)
        
        while queue:
            u = queue.pop(0)
            for v, weight in self.adj_list[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)
        
        all_nodes = set(range(self.num_vertices))
        unreachable = list(all_nodes - visited)
        unreachable.sort()
        
        return (len(unreachable) == 0), unreachable
