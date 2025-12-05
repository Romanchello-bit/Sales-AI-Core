import random
import time
import statistics
from graph_module import Graph
from algorithms import bellman_ford_list, bellman_ford_matrix

# --- SCIENTIFIC CORE ---

def generate_erdos_renyi(n, density):
    """
    Generates a random directed graph using the Erdős-Rényi model.
    
    Args:
        n (int): Number of vertices.
        density (float): Probability of edge creation (0.0 to 1.0).
        
    Returns:
        Graph: A generated graph object.
    """
    graph = Graph(n, directed=True)
    
    # Directed graph max edges = n * (n - 1) (no self-loops)
    max_edges = n * (n - 1)
    target_edges = int(max_edges * density)
    
    # Track existing edges to avoid duplicates
    existing_edges = set()
    count = 0
    
    # Safety: if target_edges > possible edges, clamp it
    if target_edges > max_edges:
        target_edges = max_edges
        
    while count < target_edges:
        u = random.randint(0, n - 1)
        v = random.randint(0, n - 1)
        
        if u != v: # No self-loops
            edge_key = (u, v)
            if edge_key not in existing_edges:
                existing_edges.add(edge_key)
                weight = random.randint(-2, 10) # Scientific constraint: -2 to 10
                graph.add_edge(u, v, weight)
                count += 1
                
    return graph

def run_scientific_benchmark(sizes=[20, 201, 20], densities=[0.1, 0.3, 0.5, 0.7, 0.9], num_runs=20):
    """
    Runs rigorous benchmarks comparing Bellman-Ford (List) vs (Matrix).
    
    Args:
        sizes (list[int]): List of vertex counts.
        densities (list[float]): List of edge densities.
        num_runs (int): Number of repetitions for averaging.
        
    Returns:
        list[dict]: List of result dictionaries ready for DataFrame.
    """
    results = []
    
    # Ensure range if sizes was passed as default range-like logic (though we expect list)
    # If the caller passes a list, use it. If the caller passes nothing, use default list.
    # The prompt said: sizes: range(20, 201, 20). 
    # Python defaults evaluated at definition, so we can't put range object directly effectively if mutable.
    # We will assume input is a list or handle the default logic here if None.
    
    # If users rely on default argument binding:
    # prompt: sizes: range(20, 201, 20) -> [20, 40, ..., 200]
    if sizes == [20, 201, 20]: # Check if it's the signature default we set (which is weird, let's fix it for safety)
         sizes = list(range(20, 201, 20))
         
    print(f"--- Scientific Benchmark Started (Runs per config: {num_runs}) ---")
    
    for n in sizes:
        for d in densities:
            time_list_accum = 0.0
            time_matrix_accum = 0.0
            
            for _ in range(num_runs):
                # a. Generate Graph
                graph = generate_erdos_renyi(n, d)
                
                # b. Convert to Matrix
                matrix = graph.to_adjacency_matrix()
                
                # c. Measure time for List
                start_l = time.perf_counter()
                try:
                    bellman_ford_list(graph, 0)
                except: pass
                end_l = time.perf_counter()
                time_list_accum += (end_l - start_l)
                
                # d. Measure time for Matrix
                start_m = time.perf_counter()
                try:
                    bellman_ford_matrix(matrix, 0)
                except: pass
                end_m = time.perf_counter()
                time_matrix_accum += (end_m - start_m)
            
            # Calculate Averages
            avg_list = time_list_accum / num_runs
            avg_matrix = time_matrix_accum / num_runs
            
            # Record Data
            results.append({
                "Vertices (N)": n,
                "Density": d,
                "Time_List": avg_list,
                "Time_Matrix": avg_matrix
            })
            print(f"N={n}, D={d} -> List: {avg_list:.6f}s | Matrix: {avg_matrix:.6f}s")
            
    return results

# --- BACKWARD COMPATIBILITY (Aliases) ---

def generate_random_graph(n, density):
    """Alias for generate_erdos_renyi to support legacy code."""
    return generate_erdos_renyi(n, density)

def run_benchmark(sizes, density):
    """
    Alias supporting legacy list[tuple] return format.
    Uses run_scientific_benchmark internally with 1 run for speed.
    """
    # Just run 1 run for speed, take the time_list as default behavior
    raw_results = run_scientific_benchmark(sizes, [density], num_runs=1)
    
    # Convert to list of tuples [(N, Time)] using Time_List (standard behavior)
    return [(r["Vertices (N)"], r["Time_List"]) for r in raw_results]

if __name__ == "__main__":
    # Test
    res = run_scientific_benchmark([10, 50], [0.1, 0.5], num_runs=5)
    print(res)
