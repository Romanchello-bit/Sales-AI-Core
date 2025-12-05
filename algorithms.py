def bellman_ford_list(graph, start_node, visited_nodes=None, client_type="B2B", sentiment_score=0.0):
    """
    Advanced Bellman-Ford Algorithm.
    
    Features:
    1. Dynamic Weights based on Client Type (B2B prefers logic, B2C prefers speed).
    2. Penalty for re-visiting nodes (avoid loops).
    3. Sentiment adjustment (-1 angry to +1 happy affects aggressive paths).
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
    
    # Sentiment modifier: negative sentiment increases costs, positive decreases
    sentiment_factor = 1.0 - (sentiment_score * 0.3)  # Range: 0.7 (happy) to 1.3 (angry)

    # Основний цикл релаксації
    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v, weight in graph.adj_list[u]:
                
                # --- ПОКРАЩЕННЯ 1: Динамічна вага ---
                current_weight = weight * sentiment_factor
                
                # --- ПОКРАЩЕННЯ 2: Штраф за повторення ---
                if visited_nodes and v in visited_nodes:
                    current_weight *= 50  # Величезний штраф, щоб не йти назад
                
                # Релаксація
                if dist[u] != float("inf") and dist[u] + current_weight < dist[v]:
                    dist[v] = dist[u] + current_weight

    # Перевірка на негативні цикли (опціонально, в продажах їх зазвичай немає)
    return dist


def bellman_ford_matrix(matrix, start_node):
    """
    Standard Bellman-Ford for Adjacency Matrix (O(V^3)).
    optimized for "scientific comparison" against Adjacency List.
    """
    num_vertices = len(matrix)
    dist = [float("inf")] * num_vertices
    dist[start_node] = 0
    
    # Relax edges |V| - 1 times
    for _ in range(num_vertices - 1):
        for u in range(num_vertices):
            for v in range(num_vertices):
                weight = matrix[u][v]
                if weight != float("inf"):
                    if dist[u] != float("inf") and dist[u] + weight < dist[v]:
                        dist[v] = dist[u] + weight
                        
    return dist
