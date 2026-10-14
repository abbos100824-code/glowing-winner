"""
SynapseDSL — Sparse Neuromorphic Graph vs. Dense Matrix Simulation PoC
Demonstrates compute (MAC/FLOP) reduction via asynchronous threshold routing.
"""

import random
import math

class NeuromorphicGraphSim:
    def __init__(self, num_nodes=512, connectivity=0.10, threshold=0.65, leak=0.90):
        self.num_nodes = num_nodes
        self.threshold = threshold
        self.leak = leak
        self.potentials = [0.0] * num_nodes
        self.refractory = [0] * num_nodes
        
        # Compressed Sparse adjacency list: node_i -> list of (target_j, weight)
        self.synapses = {i: [] for i in range(num_nodes)}
        random.seed(42)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < connectivity:
                    self.synapses[i].append((j, random.uniform(0.1, 0.45)))

    def run_simulation(self, steps=100, input_sparsity=0.08):
        dense_ops = 0
        sparse_event_ops = 0
        total_spikes = 0

        for step in range(steps):
            # Dense equivalent: full N x N matrix-vector multiplication every step
            dense_ops += self.num_nodes * self.num_nodes

            # Sparse event-driven execution: only process active incoming pulses
            active_pulses = [
                i for i in range(self.num_nodes) if random.random() < input_sparsity
            ]

            next_queue = []
            for node_id in active_pulses:
                if self.refractory[node_id] > 0:
                    self.refractory[node_id] -= 1
                    continue

                self.potentials[node_id] = (self.potentials[node_id] * self.leak) + 0.40
                sparse_event_ops += 1

                if self.potentials[node_id] >= self.threshold:
                    self.potentials[node_id] = 0.0
                    self.refractory[node_id] = 2
                    total_spikes += 1
                    next_queue.append(node_id)

            # Propagate pulses ONLY from nodes that crossed threshold
            for firing_node in next_queue:
                for target_node, weight in self.synapses[firing_node]:
                    self.potentials[target_node] += weight
                    sparse_event_ops += 1

        reduction = (1.0 - (sparse_event_ops / dense_ops)) * 100.0
        return dense_ops, sparse_event_ops, total_spikes, reduction


if __name__ == "__main__":
    print("=" * 65)
    print(" SynapseDSL — Neuromorphic Graph Compute Benchmark (PoC v0.1)")
    print("=" * 65)
    
    sim = NeuromorphicGraphSim(num_nodes=512, connectivity=0.10)
    dense, sparse, spikes, saved = sim.run_simulation(steps=100)

    print(f"Nodes in Graph         : 512 (10% sparse synaptic density)")
    print(f"Simulation Time Steps  : 100 ticks")
    print("-" * 65)
    print(f"Standard Dense Ops     : {dense:,} MAC operations")
    print(f"SynapseDSL Sparse Ops  : {sparse:,} Event operations")
    print(f"Total Neural Spikes    : {spikes:,} asynchronous events")
    print("-" * 65)
    print(f"COMPUTE SAVED          : {saved:.2f}% fewer operations!")
    print("=" * 65)
