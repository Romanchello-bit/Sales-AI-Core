import json
from graph_module import Graph
from algorithms import bellman_ford_list


class SalesEngine:
    def __init__(self, json_file='sales_script.json'):
        """Initialize the SalesEngine by loading the sales script from JSON."""
        # Load the sales script
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        self.nodes_data = data['nodes']
        self.edges_data = data['edges']
        
        # Create mapping from string IDs to integer IDs
        self.str_to_int = {}
        self.int_to_str = {}
        self.node_text = {}
        
        for idx, node_name in enumerate(self.nodes_data.keys()):
            self.str_to_int[node_name] = idx
            self.int_to_str[idx] = node_name
            self.node_text[node_name] = self.nodes_data[node_name]
        
        # Build the Graph object
        num_nodes = len(self.nodes_data)
        self.graph = Graph(num_nodes, directed=True)
        
        # Add edges with weights
        for edge in self.edges_data:
            from_node = self.str_to_int[edge['from']]
            to_node = self.str_to_int[edge['to']]
            weight = edge['weight']
            self.graph.add_edge(from_node, to_node, weight)
    
    def get_best_next_step(self, current_step_name):
        """
        Find the best next step from the current position to reach 'close_deal'.
        
        Args:
            current_step_name: String name of the current step
            
        Returns:
            Tuple of (next_step_name, next_step_text) for the optimal next step
        """
        # Find integer ID of current step
        current_id = self.str_to_int[current_step_name]
        
        # Run Bellman-Ford from current step
        distances = bellman_ford_list(self.graph, current_id)
        
        if distances is None:
            raise ValueError("Negative cycle detected in the sales script graph!")
        
        # Find the close_deal node ID
        close_deal_id = self.str_to_int['close_deal']
        
        # If we're already at close_deal, return it
        if current_id == close_deal_id:
            return current_step_name, self.node_text[current_step_name]
        
        # Find the best immediate next step
        # Look at all neighbors of current node and pick the one with the shortest total distance to close_deal
        adj_list = self.graph.get_list()
        best_next_id = None
        best_total_distance = float('inf')
        
        for neighbor_id, edge_weight in adj_list[current_id]:
            # Compute distance from this neighbor to close_deal
            neighbor_distances = bellman_ford_list(self.graph, neighbor_id)
            to_close = neighbor_distances[close_deal_id]
            total_distance = edge_weight + to_close
            if total_distance < best_total_distance:
                best_total_distance = total_distance
                best_next_id = neighbor_id
        
        if best_next_id is None:
            raise ValueError(f"No path found from '{current_step_name}' to 'close_deal'")
        
        # Convert back to string name and get text
        next_step_name = self.int_to_str[best_next_id]
        next_step_text = self.node_text[next_step_name]
        
        return next_step_name, next_step_text


if __name__ == "__main__":
    # Simulate a conversation path
    engine = SalesEngine()
    
    current_step = "start"
    print(f"Starting sales conversation at: {current_step}")
    print(f">>> {engine.node_text[current_step]}\n")
    
    step_count = 0
    max_steps = 10  # Safety limit to prevent infinite loops
    
    while current_step != "close_deal" and step_count < max_steps:
        # Get the best next step
        next_step, next_text = engine.get_best_next_step(current_step)
        
        print(f"Best next move: {next_step}")
        print(f">>> {next_text}\n")
        
        current_step = next_step
        step_count += 1
    
    if current_step == "close_deal":
        print("[SUCCESS] Successfully reached the deal closure!")
    else:
        print("[WARNING] Reached maximum steps without closing the deal.")
