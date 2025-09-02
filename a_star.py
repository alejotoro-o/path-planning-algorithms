import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- Node class ---
class Node:
    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position
        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)

# --- A* implementation ---
def astar(maze, start, end):
    start_node = Node(None, start)
    end_node = Node(None, end)

    open_list = [start_node]
    closed_list = set()
    explored = []

    while open_list:
        # Get node with lowest f
        current_node = min(open_list, key=lambda node: node.f)
        open_list.remove(current_node)
        closed_list.add(current_node)

        explored.append(current_node.position)

        if current_node == end_node:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1], explored

        # Adjacent moves (8 directions)
        for new_pos in [(0,-1),(0,1),(-1,0),(1,0),(-1,-1),(-1,1),(1,-1),(1,1)]:
            node_pos = (current_node.position[0] + new_pos[0],
                        current_node.position[1] + new_pos[1])

            if not (0 <= node_pos[0] < maze.shape[0] and 0 <= node_pos[1] < maze.shape[1]):
                continue
            if maze[node_pos[0], node_pos[1]] != 0:
                continue

            child = Node(current_node, node_pos)
            if child in closed_list:
                continue

            # Costs
            cost = np.linalg.norm(np.array(node_pos) - np.array(current_node.position))
            child.g = current_node.g + cost
            child.h = abs(end_node.position[0] - child.position[0]) + abs(end_node.position[1] - child.position[1])
            child.f = child.g + child.h

            if child in open_list:
                i = open_list.index(child)
                if child.g >= open_list[i].g:
                    continue

            open_list.append(child)

    return None, explored

# --- Simulation setup ---
maze = np.zeros((20, 20))

# Vertical wall with a gap
maze[5:15, 10] = 1
maze[10, 10] = 0  # create a gap in the middle

# Horizontal wall with a gap
maze[12, 3:17] = 1
maze[12, 8] = 0  # gap

# Some random block obstacles
maze[3:5, 3:5] = 1
maze[7:9, 15:18] = 1
maze[16:18, 5:7] = 1

start = (0, 0)
end = (19, 19)

path, explored = astar(maze, start, end)

# --- Visualization ---
fig, ax = plt.subplots()
ax.set_xticks([])
ax.set_yticks([])
ax.imshow(maze, cmap="gray_r")

explored_scatter, = ax.plot([], [], "o", color="blue", markersize=4, alpha=0.5)
path_line, = ax.plot([], [], "-r", linewidth=2)

def update(frame):
    # Draw explored nodes
    if explored and frame < len(explored):
        xs, ys = zip(*explored[:frame+1])
        explored_scatter.set_data(ys, xs)
    # Draw final path after exploration ends
    if path and frame >= len(explored):
        xs, ys = zip(*path)
        path_line.set_data(ys, xs)
    return explored_scatter, path_line

def init():
    explored_scatter.set_data([], [])
    path_line.set_data([], [])
    return explored_scatter, path_line

ani = animation.FuncAnimation(fig, update, frames=len(explored)+20, interval=100, blit=True, repeat=False, init_func=init)
plt.show()

ani.save("a_star.gif", writer="pillow", fps=20)
