import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random

# --- RRT* Parameters ---
MAX_NODES = 3000
STEP_SIZE = 2.0
GOAL_SAMPLE_RATE = 0.2
GOAL_THRESHOLD = 3.0
NEIGHBOR_RADIUS = 3.0

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


# --- RRT* Implementation ---
def rrt_star(maze, start, goal, max_nodes=MAX_NODES):
    tree = {start: None}
    cost = {start: 0.0}
    nodes = [start]
    explored = []

    for i in range(max_nodes):
        # Sample random point or goal
        if random.random() < GOAL_SAMPLE_RATE:
            sample = goal
        else:
            sample = (random.uniform(0, maze.shape[0]-1),
                      random.uniform(0, maze.shape[1]-1))

        # Nearest node
        nearest = min(nodes, key=lambda n: distance(n, sample))

        # Steer
        new_node = steer(nearest, sample, STEP_SIZE)

        # Skip if already exists
        if any(distance(new_node, n) < 1e-6 for n in nodes):
            continue

        # Collision check
        if not edge_collision_free(nearest, new_node, maze):
            continue

        # --- Choose parent (RRT* improvement) ---
        neighbors = [n for n in nodes if distance(n, new_node) < NEIGHBOR_RADIUS
                     and edge_collision_free(n, new_node, maze)]
        if not neighbors:
            parent = nearest
        else:
            parent = min(neighbors, key=lambda n: cost[n] + distance(n, new_node))

        # Add node
        tree[new_node] = parent
        cost[new_node] = cost[parent] + distance(parent, new_node)
        nodes.append(new_node)
        explored.append((parent, new_node))

        # --- Rewiring step ---
        for n in neighbors:
            new_cost = cost[new_node] + distance(new_node, n)
            if new_cost < cost[n] and edge_collision_free(new_node, n, maze):
                tree[n] = new_node
                cost[n] = new_cost
                explored.append((new_node, n))

        # Goal check
        if distance(new_node, goal) < GOAL_THRESHOLD:
            if edge_collision_free(new_node, goal, maze):  # ✅ ensure final edge is valid
                path = [goal]
                current = new_node
                while current is not None:
                    path.append(current)
                    current = tree[current]
                return path[::-1], explored

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

path, explored = rrt_star(maze, start, goal)

# --- Visualization ---
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(maze, cmap="gray_r")

explored_lines = []
path_line, = ax.plot([], [], "-r", linewidth=2)

def update(frame):
    if frame < len(explored):
        n1, n2 = explored[frame]
        xs, ys = zip(n1, n2)
        line, = ax.plot(ys, xs, color="blue", alpha=0.3, linewidth=1)
        explored_lines.append(line)

    if path and frame >= len(explored):
        xs, ys = zip(*path)
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
    fig, update, frames=len(explored) + 20,
    interval=50, blit=True, repeat=False, init_func=init
)
plt.show()

ani.save("rrt_star.gif", writer="pillow", fps=20)
