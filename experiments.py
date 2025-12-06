import time
import random
import pandas as pd
from graph_module import Graph
from algorithms import bellman_ford_list, bellman_ford_matrix

def generate_erdos_renyi(n, p):
    """
    Generates a random graph using the Erdős-Rényi model.
    
    Args:
        n (int): Number of vertices.
        p (float): Probability of an edge between any two vertices.
        
    Returns:
        Graph: A random graph instance.
    """
    graph = Graph(n, directed=True)
    for i in range(n):
        for j in range(n):
            if i != j and random.random() < p:
                weight = random.randint(1, 20)
                graph.add_edge(i, j, weight)
    return graph

def run_scientific_benchmark(sizes, densities, num_runs=5):
    """
    Runs a benchmark comparing list vs. matrix implementations of Bellman-Ford.
    
    Args:
        sizes (list): A list of graph sizes (number of vertices).
        densities (list): A list of graph densities (edge probability).
        num_runs (int): How many times to run for each combination to get an average.
        
    Returns:
        list: A list of dictionaries with the benchmark results.
    """
    results = []
    total_experiments = len(sizes) * len(densities)
    current_experiment = 0

    print("--- Starting Scientific Benchmark ---")
    for n in sizes:
        for d in densities:
            current_experiment += 1
            print(f"Running experiment {current_experiment}/{total_experiments}: Size={n}, Density={d}")
            
            time_list_total = 0
            time_matrix_total = 0
            
            for i in range(num_runs):
                # Generate a new random graph for each run
                g = generate_erdos_renyi(n, d)
                
                # Benchmark list implementation
                start_time = time.perf_counter()
                bellman_ford_list(g, 0)
                end_time = time.perf_counter()
                time_list_total += (end_time - start_time)
                
                # Benchmark matrix implementation
                start_time = time.perf_counter()
                bellman_ford_matrix(g, 0)
                end_time = time.perf_counter()
                time_matrix_total += (end_time - start_time)
            
            avg_list = time_list_total / num_runs
            avg_matrix = time_matrix_total / num_runs
            
            results.append({
                "Vertices (N)": n,
                "Density": d,
                "Time_List": avg_list,
                "Time_Matrix": avg_matrix
            })
            
    print("--- Scientific Benchmark Finished ---")
    return results
