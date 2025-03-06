import networkx as nx
import itertools

def contains_cycle(edge_set):
    # TODO: networkx has a method to test for cycles (faster than listing all of them)
    G = nx.DiGraph()
    G.add_edges_from(edge_set)
    cycles = list(nx.simple_cycles(G))
    return len(cycles) > 0

def generate_larger_edge_sets(edge_set, edges, acyclic=False):
    larger_edge_sets = [
        x for e in edges 
        if e not in edge_set
        and not (contains_cycle((x := edge_set + [e])) and acyclic)
    ]
    return larger_edge_sets


def generate_smaller_edge_sets(edge_set):
    n = len(edge_set) - 1
    if n < 0:
        return []
    
    smaller_edge_sets = itertools.combinations(edge_set, n)
    return list(map(list, smaller_edge_sets))


def generate_all_edges(n: int):
    return  [(i, j) for i in range(1, n+1) for j in range(1, n+1) if i != j]