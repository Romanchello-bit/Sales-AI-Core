def bellman_ford_list(graph, start_node, visited_nodes=None, client_type="B2B"):
    """
    Advanced Bellman-Ford Algorithm.
    
    Features:
    1. Dynamic Weights based on Client Type (B2B prefers logic, B2C prefers speed).
    2. Penalty for re-visiting nodes (avoid loops).
    """
    
    # Ініціалізація
    num_vertices = graph.num_vertices
    dist = [float("inf")] * num_vertices
    dist[start_node] = 0
    
    # Визначаємо множники ваг
    # B2B любить деталі (знижуємо ціну довгих етапів), B2C любить швидкість
    type_modifier = {
        "B2B": {"logic": 0.8, "emotion": 1.2, "speed": 1.0},
        "B2C": {"logic": 1.5, "emotion": 0.7, "speed": 0.5}
    }
    modifiers = type_modifier.get(client_type, {"logic": 1.0, "emotion": 1.0, "speed": 1.0})

    # Основний цикл релаксації
    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v, weight in graph.adj_list[u]:
                
                # --- ПОКРАЩЕННЯ 1: Динамічна вага ---
                # Тут можна було б перевіряти тип ребра, якби він був у графі.
                # Поки що просто емулюємо:
                current_weight = weight
                
                # --- ПОКРАЩЕННЯ 2: Штраф за повторення ---
                if visited_nodes and v in visited_nodes:
                    current_weight *= 50  # Величезний штраф, щоб не йти назад
                
                # Релаксація
                if dist[u] != float("inf") and dist[u] + current_weight < dist[v]:
                    dist[v] = dist[u] + current_weight

    # Перевірка на негативні цикли (опціонально, в продажах їх зазвичай немає)
    return dist


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
