def bellman_ford_list(graph, start_node):
    distances = {i: float('inf') for i in range(graph.num_vertices)}
    distances[start_node] = 0
    adj_list = graph.get_list()
    
    # Relaxation steps
    for _ in range(graph.num_vertices - 1):
        changed = False
        for u in adj_list:
            for v, weight in adj_list[u]:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    changed = True
        if not changed:
            break
            
    # Negative cycle detection
    for u in adj_list:
        for v, weight in adj_list[u]:
            if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                return None # Negative cycle detected

    return distances

def bellman_ford_matrix(graph, start_node):
    distances = {i: float('inf') for i in range(graph.num_vertices)}
    distances[start_node] = 0
    matrix = graph.get_matrix()
    n = graph.num_vertices
    
    # Relaxation steps
    for _ in range(n - 1):
        changed = False
        for u in range(n):
            for v in range(n):
                weight = matrix[u][v]
                if weight is not None:
                    if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                        distances[v] = distances[u] + weight
                        changed = True
        if not changed:
            break
            
    # Negative cycle detection
    for u in range(n):
        for v in range(n):
            weight = matrix[u][v]
            if weight is not None:
                if distances[u] != float('inf') and distances[u] + weight < distances[v]:
                    return None # Negative cycle detected

    return distances
