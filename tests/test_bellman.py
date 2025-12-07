import os
import sys
# Ensure project root is on sys.path so tests can import local modules
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import unittest
import math
from graph_module import Graph
from algorithms import bellman_ford_list, bellman_ford_matrix

class BellmanFordTests(unittest.TestCase):
    def assertDistancesEqual(self, a, b):
        self.assertEqual(len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)):
            if math.isinf(x) and math.isinf(y):
                continue
            # allow small float tolerance
            self.assertAlmostEqual(x, y, places=6, msg=f"Mismatch at index {i}: {x} != {y}")

    def test_simple_graph(self):
        g = Graph(3, directed=True)
        g.add_edge(0, 1, 5)
        g.add_edge(1, 2, 3)
        g.add_edge(0, 2, 10)

        d_list = bellman_ford_list(g, 0)
        matrix = g.to_adjacency_matrix()
        d_mat = bellman_ford_matrix(matrix, 0)

        self.assertDistancesEqual(d_list, d_mat)
        self.assertEqual(d_list, [0, 5, 8])

    def test_negative_edge(self):
        g = Graph(3, directed=True)
        g.add_edge(0, 1, 4)
        g.add_edge(1, 2, -2)
        g.add_edge(0, 2, 5)

        d_list = bellman_ford_list(g, 0)
        matrix = g.to_adjacency_matrix()
        d_mat = bellman_ford_matrix(matrix, 0)

        self.assertDistancesEqual(d_list, d_mat)
        self.assertEqual(d_list, [0, 4, 2])

    def test_disconnected_node(self):
        g = Graph(4, directed=True)
        g.add_edge(0, 1, 1)
        g.add_edge(1, 2, 2)
        # node 3 is isolated

        d_list = bellman_ford_list(g, 0)
        matrix = g.to_adjacency_matrix()
        d_mat = bellman_ford_matrix(matrix, 0)

        self.assertDistancesEqual(d_list, d_mat)
        self.assertTrue(math.isinf(d_list[3]), "Node 3 should be unreachable and distance should be inf")

if __name__ == '__main__':
    unittest.main()
