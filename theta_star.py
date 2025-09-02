import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import heapq

# --- Helper functions ---
def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def steer(from_node, to_point, step_size):
    vec = np.array(to_point) - np.array(from_node)
    dist = np.linalg.norm(vec)
    if dist == 0:
        return tuple(from_node)
    direction = vec / dist
    new_point = np.array(from_node) + step_size * direction
    return tuple(new_point)   # keep float for smoother exploration

def edge_collision_free(p1, p2, maze):
    """
    Robust grid collision check using a supercover Bresenham line.
    Ensures every grid cell the segment passes through is checked.
    """
    rows, cols = maze.shape
    in_bounds = lambda x, y: 0 <= x < rows and 0 <= y < cols

    x0, y0 = int(round(p1[0])), int(round(p1[1]))
    x1, y1 = int(round(p2[0])), int(round(p2[1]))

    if not in_bounds(x0, y0) or not in_bounds(x1, y1):
        return False

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x1 >= x0 else -1
    sy = 1 if y1 >= y0 else -1
    err = dx - dy

    x, y = x0, y0
    if maze[x, y] != 0:
        return False

    while (x != x1) or (y != y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
            if not in_bounds(x, y) or maze[x, y] != 0:
                return False
        if e2 < dx:
            err += dx
            y += sy
            if not in_bounds(x, y) or maze[x, y] != 0:
                return False

    return True

# --- Theta* Implementation ---
def theta_star(maze, start, goal):
    rows, cols = maze.shape
    start = (int(start[0]), int(start[1]))
    goal = (int(goal[0]), int(goal[1]))

    def in_bounds(p):
        return 0 <= p[0] < rows and 0 <= p[1] < cols

    def neighbors(p):
        # 8-connected grid
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                n = (p[0] + dx, p[1] + dy)
                if in_bounds(n) and maze[n[0], n[1]] == 0:
                    yield n

    g = {start: 0}
    parent = {start: start}
    open_set = [(distance(start, goal), start)]
    closed = set()
    explored = []

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in closed:
            continue
        closed.add(current)

        if current == goal:
            # Reconstruct path
            path = [goal]
            while path[-1] != start:
                path.append(parent[path[-1]])
            return path[::-1], explored

        for n in neighbors(current):
            if n in closed:
                continue
            # Try line-of-sight optimization
            if parent[current] and edge_collision_free(parent[current], n, maze):
                new_g = g[parent[current]] + distance(parent[current], n)
                if new_g < g.get(n, float("inf")):
                    g[n] = new_g
                    parent[n] = parent[current]
                    f = new_g + distance(n, goal)
                    heapq.heappush(open_set, (f, n))
                    explored.append((parent[n], n))
            else:
                new_g = g[current] + distance(current, n)
                if new_g < g.get(n, float("inf")):
                    g[n] = new_g
                    parent[n] = current
                    f = new_g + distance(n, goal)
                    heapq.heappush(open_set, (f, n))
                    explored.append((parent[n], n))

    return None, explored

# --- Maze Setup ---
maze = np.zeros((20, 20))
maze[5:15, 10] = 1
maze[10, 10] = 0
maze[12, 3:17] = 1
maze[12, 8] = 0
maze[3:5, 3:5] = 1
maze[7:9, 15:18] = 1
maze[16:18, 5:7] = 1

start = (0.0, 0.0)
goal = (19.0, 19.0)

# --- Run Theta* ---
theta_path, theta_explored = theta_star(maze, start, goal)

# --- Visualization for Theta* ---
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(maze, cmap="gray_r")

explored_lines = []
path_line, = ax.plot([], [], "-r", linewidth=2)

def update(frame):
    if frame < len(theta_explored):
        n1, n2 = theta_explored[frame]
        xs, ys = zip(n1, n2)
        line, = ax.plot(ys, xs, color="blue", alpha=0.3, linewidth=1)
        explored_lines.append(line)

    if theta_path and frame >= len(theta_explored):
        xs, ys = zip(*theta_path)
        path_line.set_data(ys, xs)

    return explored_lines + [path_line]

def init():
    # Clear previously drawn edges
    for line in explored_lines:
        line.remove()
    explored_lines.clear()

    # Reset the path line
    path_line.set_data([], [])
    return [path_line]

ani = animation.FuncAnimation(
    fig, update, frames=len(theta_explored) + 20,
    interval=50, blit=True, repeat=False, init_func=init
)
plt.show()

ani.save("theta_star.gif", writer="pillow", fps=20)
