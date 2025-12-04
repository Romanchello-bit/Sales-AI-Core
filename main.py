import time
import matplotlib.pyplot as plt
from generator import generate_random_graph
from algorithms import bellman_ford_list, bellman_ford_matrix

def run_experiments():
    sizes = [20, 50, 100, 200]
    densities = [0.2, 0.5, 0.8]
    results = {d: {'list': [], 'matrix': []} for d in densities}

    print(f"{'Size':<10} {'Density':<10} {'List (s)':<15} {'Matrix (s)':<15}")
    print("-" * 55)
    
    for d in densities:
        for n in sizes:
            t_list, t_matrix = 0, 0
            runs = 5
            for _ in range(runs):
                g = generate_random_graph(n, d)
                
                start = time.perf_counter()
                bellman_ford_list(g, 0)
                t_list += (time.perf_counter() - start)
                
                start = time.perf_counter()
                bellman_ford_matrix(g, 0)
                t_matrix += (time.perf_counter() - start)
            
            avg_list = t_list / runs
            avg_matrix = t_matrix / runs
            results[d]['list'].append(avg_list)
            results[d]['matrix'].append(avg_matrix)
            print(f"{n:<10} {d:<10} {avg_list:.6f}        {avg_matrix:.6f}")

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for d in densities:
        ax1.plot(sizes, results[d]['list'], marker='o', label=f'D={d}')
    ax1.set_title("Bellman-Ford (List)")
    ax1.set_xlabel("Number of Vertices")
    ax1.set_ylabel("Time (seconds)")
    ax1.legend()
    ax1.grid(True)
    
    for d in densities:
        ax2.plot(sizes, results[d]['matrix'], marker='o', label=f'D={d}')
    ax2.set_title("Bellman-Ford (Matrix)")
    ax2.set_xlabel("Number of Vertices")
    ax2.set_ylabel("Time (seconds)")
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('benchmark_results.png')
    print("\nBenchmark finished. Results saved to benchmark_results.png")
    plt.show()

if __name__ == "__main__":
    run_experiments()
