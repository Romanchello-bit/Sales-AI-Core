import random
import time
import statistics
from graph_module import Graph
from algorithms import bellman_ford_list

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

def run_scientific_benchmark(sizes, densities, num_runs=20):
    """
    Runs rigorous benchmarks for Bellman-Ford algorithm.
    CRITICAL: For each pair (n, d), we repeat the experiment num_runs times.
    
    Args:
        sizes (list[int]): List of vertex counts.
        densities (list[float]): List of edge densities.
        num_runs (int): Number of repetitions for averaging.
        
    Returns:
        list[dict]: List of result dictionaries ready for DataFrame.
    """
    results = []
    
    print(f"--- Scientific Benchmark Started (Runs per config: {num_runs}) ---")
    
    for n in sizes:
        for d in densities:
            times = []
            edge_counts = []
            
            for _ in range(num_runs):
                # 1. Generate NEW random graph (don't count this in timing)
                graph = generate_erdos_renyi(n, d)
                
                # Count actual edges
                # Sum of adjacency list lengths
                e_count = sum(len(graph.adj_list[u]) for u in range(graph.num_vertices))
                edge_counts.append(e_count)
                
                # 2. Start Timer
                start_time = time.perf_counter()
                
                # 3. Run Algorithm (Source = 0)
                try:
                    bellman_ford_list(graph, 0)
                except Exception:
                    pass # Ignore errors (e.g. negative cycles in random graphs)
                
                # 4. Stop Timer
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            # Calculate Averages
            avg_time = statistics.mean(times)
            avg_edges = statistics.mean(edge_counts)
            
            # 5. Record Data
            results.append({
                "Vertices (N)": n,
                "Density": d,
                "Avg_Time_Sec": avg_time,
                "Edges_Count": int(avg_edges),
                "Runs": num_runs
            })
            print(f"Config N={n}, D={d} -> Avg Time: {avg_time:.6f}s")
            
    return results

# --- BACKWARD COMPATIBILITY (Aliases) ---

def generate_random_graph(n, density):
    """Alias for generate_erdos_renyi to support legacy code."""
    return generate_erdos_renyi(n, density)

def run_benchmark(sizes, density):
    """
    Alias supporting legacy list[tuple] return format.
    Uses run_scientific_benchmark internally with 1 run for speed/demo.
    """
    # Wrap density in list, run 1 time (demo mode usually needs speed)
    # The user asked for rigorous benchmark (20 runs) in the new function, 
    # but the old function was for a quick demo. 
    # Let's do 1 run to keep UI responsive, or 5 for better stability?
    # Let's stick to 1 to match previous "quick" expectation unless specified.
    
    raw_results = run_scientific_benchmark(sizes, [density], num_runs=1)
    
    # Convert to list of tuples [(N, Time)]
    return [(r["Vertices (N)"], r["Avg_Time_Sec"]) for r in raw_results]

if __name__ == "__main__":
    # Test
    res = run_scientific_benchmark([10, 50], [0.1, 0.5], num_runs=5)
    print(res)
