import math
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict
from dataclasses import dataclass
import heapq

# Data structures for the algorithms
@dataclass
class Edge:
    src: int
    dest: int
    weight: float

@dataclass
class Node:
    id: int
    x: float
    y: float
    
class DisjointSet:
    def __init__(self, vertices: int):
        self.parent = list(range(vertices))
        self.rank = [0] * vertices
    
    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])  # Path compression
        return self.parent[item]
    
    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1

# 1. Minimum Spanning Tree (Kruskal's Algorithm)
def kruskal_mst(vertices: int, edges: List[Edge]) -> List[Edge]:
    """
    Implements Kruskal's algorithm for finding Minimum Spanning Tree
    Time Complexity: O(E log E) where E is number of edges
    Space Complexity: O(V) where V is number of vertices
    """
    # Sort edges by weight
    sorted_edges = sorted(edges, key=lambda x: x.weight)
    disjoint_set = DisjointSet(vertices)
    mst = []
    
    for edge in sorted_edges:
        if len(mst) == vertices - 1:  # MST has V-1 edges
            break
            
        if disjoint_set.find(edge.src) != disjoint_set.find(edge.dest):
            disjoint_set.union(edge.src, edge.dest)
            mst.append(edge)
    
    return mst

# 2. Dijkstra's Algorithm
def dijkstra(graph: Dict[int, List[Tuple[int, float]]], start: int, end: int) -> Tuple[List[int], float]:
    """
    Implements Dijkstra's algorithm for finding shortest path
    Time Complexity: O((V + E) log V) where V is vertices and E is edges
    Space Complexity: O(V)
    """
    distances = defaultdict(lambda: float('inf'))
    distances[start] = 0
    previous = {}
    pq = [(0, start)]  # (distance, vertex)
    visited = set()
    
    while pq:
        current_distance, current = heapq.heappop(pq)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end:
            break
            
        for neighbor, weight in graph[current]:
            distance = current_distance + weight
            
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous[neighbor] = current
                heapq.heappush(pq, (distance, neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
    path.reverse()
    
    return path, distances[end]

# 3. A* Algorithm
def manhattan_distance(node1: Node, node2: Node) -> float:
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)

def a_star(graph: Dict[int, List[Tuple[int, float]]], nodes: Dict[int, Node], 
           start: int, end: int) -> Tuple[List[int], float]:
    """
    Implements A* algorithm for finding optimal path using heuristics
    Time Complexity: O((V + E) log V) in worst case, but usually better than Dijkstra
    Space Complexity: O(V)
    """
    start_node = nodes[start]
    end_node = nodes[end]
    
    g_score = defaultdict(lambda: float('inf'))
    g_score[start] = 0
    
    f_score = defaultdict(lambda: float('inf'))
    f_score[start] = manhattan_distance(start_node, end_node)
    
    open_set = [(f_score[start], start)]
    previous = {}
    visited = set()
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == end:
            break
            
        for neighbor, weight in graph[current]:
            tentative_g = g_score[current] + weight
            
            if tentative_g < g_score[neighbor]:
                previous[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + manhattan_distance(nodes[neighbor], end_node)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = end
    while current in previous:
        path.append(current)
        current = previous[current]
    path.append(start)
    path.reverse()
    
    return path, g_score[end]

# 4. Dynamic Programming for Transit Schedule Optimization
def optimize_transit_schedule(stations: List[int], demands: List[List[float]], 
                            max_time: int) -> Tuple[List[float], float]:
    """
    Optimizes transit schedule using dynamic programming
    Time Complexity: O(n * max_time) where n is number of stations
    Space Complexity: O(n * max_time)
    """
    n = len(stations)
    # dp[i][t] represents max satisfied demand up to station i at time t
    dp = [[0.0] * (max_time + 1) for _ in range(n)]
    decisions = [[0] * (max_time + 1) for _ in range(n)]
    
    # Initialize first station
    for t in range(max_time + 1):
        dp[0][t] = demands[0][min(t, len(demands[0]) - 1)]
    
    # Fill dp table
    for i in range(1, n):
        for t in range(max_time + 1):
            max_value = 0
            best_time = 0
            
            # Try different time allocations
            for prev_t in range(t + 1):
                current_demand = demands[i][min(t - prev_t, len(demands[i]) - 1)]
                if dp[i-1][prev_t] + current_demand > max_value:
                    max_value = dp[i-1][prev_t] + current_demand
                    best_time = prev_t
            
            dp[i][t] = max_value
            decisions[i][t] = best_time
    
    # Reconstruct solution
    schedule = []
    t = max_time
    for i in range(n-1, -1, -1):
        prev_t = decisions[i][t]
        schedule.append(t - prev_t)
        t = prev_t
    
    schedule.reverse()
    return schedule, dp[n-1][max_time]

# 5. Greedy Algorithm for Traffic Signal Optimization
def optimize_traffic_signals(intersections: List[int], 
                           traffic_flows: List[List[float]]) -> List[float]:
    """
    Optimizes traffic signal timings using a greedy approach
    Time Complexity: O(n log n) where n is number of intersections
    Space Complexity: O(n)
    """
    n = len(intersections)
    signal_timings = []
    
    # Calculate total flow for each intersection
    total_flows = []
    for i in range(n):
        total_flow = sum(traffic_flows[i])
        total_flows.append((total_flow, i))
    
    # Sort intersections by total flow (descending)
    total_flows.sort(reverse=True)
    
    # Assign green time proportional to flow
    total_cycle_time = 120  # 120 seconds cycle
    min_green_time = 15    # Minimum 15 seconds of green
    remaining_time = total_cycle_time - (min_green_time * n)
    
    # Initialize with minimum green time
    timings = [min_green_time] * n
    
    # Distribute remaining time proportionally
    total_flow = sum(flow for flow, _ in total_flows)
    for flow, idx in total_flows:
        if total_flow > 0:
            extra_time = (flow / total_flow) * remaining_time
            timings[idx] += extra_time
    
    return timings

# Helper function to build graph from edges
def build_graph(edges: List[Edge]) -> Dict[int, List[Tuple[int, float]]]:
    """
    Converts edge list to adjacency list representation
    """
    graph = defaultdict(list)
    for edge in edges:
        graph[edge.src].append((edge.dest, edge.weight))
        graph[edge.dest].append((edge.src, edge.weight))  # For undirected graph
    return graph 