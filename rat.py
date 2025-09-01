import heapq
import math
from collections import deque, defaultdict

class PipeNetwork:
    def __init__(self, n):
        self.n = n
        self.graph = defaultdict(list)
        self.coords = {}

    def add_pipe(self, u, v, cost):
        self.graph[u].append((v, cost))
        self.graph[v].append((u, cost))  

    def set_coordinates(self, node, x, y):
        self.coords[node] = (x, y)

    def heuristic(self, a, b):
        (x1, y1), (x2, y2) = self.coords[a], self.coords[b]
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    def dfs(self, start, goal):
        stack = [(start, [start], 0)]
        visited = set()
        while stack:
            node, path, cost = stack.pop()
            if node == goal:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.graph[node]:
                    stack.append((neighbor, path + [neighbor], cost + c))
        return None
    def bfs(self, start, goal):
        queue = deque([(start, [start], 0)])
        visited = set()
        while queue:
            node, path, cost = queue.popleft()
            if node == goal:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.graph[node]:
                    queue.append((neighbor, path + [neighbor], cost + c))
        return None
    def dijkstra(self, start, goal):
        pq = [(0, start, [start])]
        visited = set()
        while pq:
            cost, node, path = heapq.heappop(pq)
            if node == goal:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.graph[node]:
                    heapq.heappush(pq, (cost + c, neighbor, path + [neighbor]))
        return None
    def astar(self, start, goal):
        pq = [(self.heuristic(start, goal), 0, start, [start])]
        visited = set()
        while pq:
            est_total, cost, node, path = heapq.heappop(pq)
            if node == goal:
                return path, cost, len(visited)
            if node not in visited:
                visited.add(node)
                for neighbor, c in self.graph[node]:
                    new_cost = cost + c
                    est = new_cost + self.heuristic(neighbor, goal)
                    heapq.heappush(pq, (est, new_cost, neighbor, path + [neighbor]))
        return None
if __name__ == "__main__":
    n = 5
    network = PipeNetwork(n)
    coords = {0: (0,0), 1: (1,0), 2: (1,1), 3: (2,1), 4: (2,2)}
    for node, (x, y) in coords.items():
        network.set_coordinates(node, x, y)

    # Add pipes (u, v, cost)
    pipes = [(0,1,2), (1,2,2), (0,2,4), (2,3,1), (3,4,3)]
    for u, v, c in pipes:
        network.add_pipe(u, v, c)

    start, goal = 0, 4

    print("DFS:", network.dfs(start, goal))
    print("BFS:", network.bfs(start, goal))
    print("Dijkstra:", network.dijkstra(start, goal))
    print("A*:", network.astar(start, goal))
