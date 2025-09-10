import heapq
import math
import time

# Define directions for 4-way and 8-way movement
DIRECTIONS_4 = [(-1,0), (1,0), (0,-1), (0,1)]
DIRECTIONS_8 = [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]

# Heuristic functions
def manhattan(a, b):
    return abs(a[0]-b[0]) + abs(a[1]-b[1])

def euclidean(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def diagonal(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy)

# Parse the grid and find start and goal
def parse_grid(grid):
    start = goal = None
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            if cell == 'S':
                start = (i, j)
            elif cell == 'G':
                goal = (i, j)
    return start, goal

# Check if a cell is walkable
def is_valid(grid, x, y):
    rows, cols = len(grid), len(grid[0])
    return 0 <= x < rows and 0 <= y < cols and grid[x][y] != '1'

# Get cost of a cell (for A*), ghost zones cost more
def get_cost(grid, x, y):
    if grid[x][y] == '2':
        return 5  # High cost
    else:
        return 1

# Reconstruct path
def reconstruct_path(came_from, end):
    path = []
    while end in came_from:
        path.append(end)
        end = came_from[end]
    path.reverse()
    return path

# Greedy Best-First Search
def greedy_bfs(grid, start, goal, heuristic, diagonal=False):
    dirs = DIRECTIONS_8 if diagonal else DIRECTIONS_4
    visited = set()
    came_from = {}
    frontier = []
    heapq.heappush(frontier, (heuristic(start, goal), start))
    nodes_explored = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_explored += 1

        if current == goal:
            return reconstruct_path(came_from, current), nodes_explored

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in dirs:
            nx, ny = current[0] + dx, current[1] + dy
            if is_valid(grid, nx, ny) and (nx, ny) not in visited:
                came_from[(nx, ny)] = current
                heapq.heappush(frontier, (heuristic((nx, ny), goal), (nx, ny)))

    return [], nodes_explored

# A* Search
def a_star(grid, start, goal, heuristic, diagonal=False):
    dirs = DIRECTIONS_8 if diagonal else DIRECTIONS_4
    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g_score = {start: 0}
    visited = set()
    nodes_explored = 0

    while open_set:
        _, cost, current = heapq.heappop(open_set)
        nodes_explored += 1

        if current == goal:
            return reconstruct_path(came_from, current), nodes_explored

        if current in visited:
            continue
        visited.add(current)

        for dx, dy in dirs:
            nx, ny = current[0] + dx, current[1] + dy
            neighbor = (nx, ny)
            if not is_valid(grid, nx, ny):
                continue

            tentative_g = g_score[current] + get_cost(grid, nx, ny)

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, tentative_g, neighbor))

    return [], nodes_explored

# Visualize the path
def print_grid_with_path(grid, path):
    new_grid = [list(row) for row in grid]
    for x, y in path:
        if new_grid[x][y] not in ('S', 'G'):
            new_grid[x][y] = '*'
    for row in new_grid:
        print(''.join(row))

# Run test
def run_test(grid, heuristic_fn, algo_name, use_a_star=True, diagonal=False):
    start, goal = parse_grid(grid)
    start_time = time.time()

    if use_a_star:
        path, explored = a_star(grid, start, goal, heuristic_fn, diagonal)
    else:
        path, explored = greedy_bfs(grid, start, goal, heuristic_fn, diagonal)

    end_time = time.time()
    print(f"\n--- {algo_name} ---")
    print_grid_with_path(grid, path)
    print(f"Path length: {len(path)}")
    print(f"Nodes explored: {explored}")
    print(f"Execution time: {end_time - start_time:.5f} seconds")
    return path, explored

# Example grid
grid = [
    "S0010",
    "1101G",
    "00010",
    "11011",
    "00000"
]

# Add a ghost zone
grid_with_ghosts = [
    "S0010",
    "1101G",
    "00210",
    "11011",
    "00000"
]

# Run all combinations
heuristics = {
    'Manhattan': manhattan,
    'Euclidean': euclidean,
    'Diagonal': diagonal
}

for name, h in heuristics.items():
    run_test(grid, h, f"Greedy Best-First ({name})", use_a_star=False)
    run_test(grid, h, f"A* Search ({name})", use_a_star=True)

# Bonus: Try ghost zones and diagonal movement
print("\n=== Bonus: With Ghost Zones and Diagonal Movement ===")
run_test(grid_with_ghosts, manhattan, "A* with Ghost Zones & Diagonals", use_a_star=True, diagonal=True)
