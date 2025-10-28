
"""
Search Visualizer — Fixed version with proper animations for all algorithms.

Key improvements:
- BFS: Arrows show horizontal level-by-level traversal from current node
- DFS: Arrows show vertical depth-first traversal from current node
- UCS: Added missing edges (G→H, D→H), correct shortest path finding
- DLS: Proper depth-first animation with depth limit line
- IDS: Shows iteration details, limit lines, and proper animation

Run: python3 search_visualizer_fixed.py
"""

import math
import tkinter as tk
from tkinter import ttk
from collections import deque
import heapq
import copy

# --------------------------
# Image files
# --------------------------
TREE1_IMG = '/mnt/data/d9eea286-9114-4fbd-802b-ed5b830911fc.png'
TREE2_IMG = '/mnt/data/e53cb032-36e8-4d1f-ac81-24b800708964.png'

# --------------------------
# Graph definitions
# --------------------------
GRAPH_TREE1 = {
    'A': [('B', 1), ('C', 1)],
    'B': [('D', 1), ('E', 1)],
    'C': [('F', 1), ('G', 1)],
    'D': [('H', 1), ('I', 1)],
    'E': [('J', 1), ('K', 1)],
    'F': [('L', 1), ('M', 1)],
    'G': [('N', 1), ('O', 1)],
    'H': [], 'I': [], 'J': [], 'K': [], 'L': [], 'M': [], 'N': [], 'O': []
}

# Fixed GRAPH_TREE2 with added edges G→H and D→H
GRAPH_TREE2 = {
    'A': [('B', 1), ('C', 1)],
    'B': [('D', 2), ('E', 1)],
    'C': [('F', 1), ('G', 3)],
    'D': [('E', 1), ('H', 5)],  # Added D→H
    'E': [('H', 2)],
    'F': [('H', 1)],
    'G': [('F', 1), ('H', 2)],  # Added G→H
    'H': []
}

POS_TREE1 = {
    'A': (420, 70),
    'B': (250, 170), 'C': (590, 170),
    'D': (130, 270), 'E': (350, 270), 'F': (510, 270), 'G': (730, 270),
    'H': (70, 390), 'I': (190, 390), 'J': (310, 390), 'K': (390, 390),
    'L': (470, 390), 'M': (590, 390), 'N': (710, 390), 'O': (830, 390)
}

POS_TREE2 = {
    'A': (300, 50),
    'B': (180, 130), 'C': (420, 130),
    'D': (100, 210), 'E': (220, 210), 'F': (380, 210), 'G': (500, 210),
    'H': (300, 300)
}

LEVEL_Y = [70, 170, 270, 390]

# --------------------------
# Utility
# --------------------------

def reconstruct_path(parent_map, node):
    path = []
    cur = node
    while cur is not None:
        path.append(cur)
        cur = parent_map.get(cur)
    return list(reversed(path))


# --------------------------
# Visualizer
# --------------------------
class SearchVisualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Search Visualizer — Fixed Version")

        # Try load images
        try:
            self.tree1_img = tk.PhotoImage(file=TREE1_IMG)
        except Exception:
            self.tree1_img = None
        try:
            self.tree2_img = tk.PhotoImage(file=TREE2_IMG)
        except Exception:
            self.tree2_img = None

        self.graph = GRAPH_TREE1
        self.node_positions = copy.deepcopy(POS_TREE1)
        self.node_radius = 24

        # animation state
        self.running = False
        self.timer_id = None
        self.step_index = 0
        self.steps = []
        self.explored_order = []
        self.expanded_edges = []

        # dashed-arrow animation offset (for moving dashes)
        self.dash_offset = 0

        self.node_items = {}

        self.setup_ui()
        self.draw_graph()

    def setup_ui(self):
        # Controls at top
        ctrl = tk.Frame(self.root, bg='#f5f5f5', padx=8, pady=6)
        ctrl.pack(fill='x')

        tk.Label(ctrl, text="Algorithm:", bg='#f5f5f5', font=('Arial', 10, 'bold')).grid(row=0, column=0, padx=4)
        self.algo_var = tk.StringVar(value="BFS")
        algo_combo = ttk.Combobox(ctrl, textvariable=self.algo_var,
                                  values=["BFS", "UCS", "DFS", "DLS", "IDS"],
                                  state="readonly", width=8)
        algo_combo.grid(row=0, column=1, padx=4)
        algo_combo.bind('<<ComboboxSelected>>', lambda e: self.on_algo_change())

        tk.Label(ctrl, text="Start:", bg='#f5f5f5', font=('Arial', 10, 'bold')).grid(row=0, column=2, padx=4)
        self.start_var = tk.StringVar(value="A")
        start_combo = ttk.Combobox(ctrl, textvariable=self.start_var,
                                   values=sorted(list(set(list(GRAPH_TREE1.keys()) + list(GRAPH_TREE2.keys())))),
                                   state="readonly", width=6)
        start_combo.grid(row=0, column=3, padx=4)
        start_combo.bind('<<ComboboxSelected>>', lambda e: self.on_start_change())

        tk.Label(ctrl, text="Goal:", bg='#f5f5f5', font=('Arial', 10, 'bold')).grid(row=0, column=4, padx=4)
        self.goal_var = tk.StringVar(value="G")
        ttk.Combobox(ctrl, textvariable=self.goal_var,
                     values=sorted(list(set(list(GRAPH_TREE1.keys()) + list(GRAPH_TREE2.keys())))), state="readonly", width=6).grid(row=0, column=5, padx=4)

        # DLS/IDS controls
        self.depth_lbl = tk.Label(ctrl, text="Depth Limit:", bg='#f5f5f5', font=('Arial', 10, 'bold'))
        self.depth_var = tk.IntVar(value=2)
        self.depth_spin = tk.Spinbox(ctrl, from_=0, to=10, textvariable=self.depth_var, width=5)

        self.iter_lbl = tk.Label(ctrl, text="Max Iter:", bg='#f5f5f5', font=('Arial', 10, 'bold'))
        self.iter_var = tk.IntVar(value=3)
        self.iter_spin = tk.Spinbox(ctrl, from_=0, to=10, textvariable=self.iter_var, width=5)

        # Speed
        tk.Label(ctrl, text="Speed (ms):", bg='#f5f5f5', font=('Arial', 10)).grid(row=1, column=0, padx=4)
        self.speed_var = tk.IntVar(value=300)
        tk.Scale(ctrl, from_=50, to=1000, orient='horizontal', variable=self.speed_var, length=240,
                 bg='#f5f5f5').grid(row=1, column=1, columnspan=2, sticky='w')

        tk.Button(ctrl, text="▶ Run Search", command=self.run_search,
                  bg='#4CAF50', fg='white', font=('Arial', 11, 'bold'), width=12).grid(row=1, column=3, padx=6)
        tk.Button(ctrl, text="⟲ Reset", command=self.reset,
                  bg='#f44336', fg='white', font=('Arial', 11, 'bold'), width=10).grid(row=1, column=4, padx=6)

        # Main content area: canvas left, info right
        main = tk.Frame(self.root)
        main.pack(fill='both', expand=True, padx=10, pady=8)

        # Canvas frame (left)
        canvas_fr = tk.Frame(main)
        canvas_fr.pack(side='left', fill='both', expand=True)
        self.canvas = tk.Canvas(canvas_fr, bg='white', highlightthickness=2, highlightbackground='#ccc')
        self.canvas.pack(fill='both', expand=True)

        # Info frame (right) - with scrollbar for IDS iterations
        info_frame = tk.Frame(main, bg='#e3f2fd')
        info_frame.pack(side='right', fill='both')
        
        # Create a canvas with scrollbar for the info panel
        info_canvas = tk.Canvas(info_frame, bg='#e3f2fd', width=300, highlightthickness=0)
        scrollbar = ttk.Scrollbar(info_frame, orient="vertical", command=info_canvas.yview)
        self.scrollable_info = tk.Frame(info_canvas, bg='#e3f2fd', padx=10, pady=8)
        
        self.scrollable_info.bind(
            "<Configure>",
            lambda e: info_canvas.configure(scrollregion=info_canvas.bbox("all"))
        )
        
        info_canvas.create_window((0, 0), window=self.scrollable_info, anchor="nw")
        info_canvas.configure(yscrollcommand=scrollbar.set)
        
        info_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Info labels in scrollable frame
        self.step_lbl = tk.Label(self.scrollable_info, text="Step: 0", font=('Arial', 12, 'bold'), bg='#e3f2fd')
        self.step_lbl.pack(pady=(4, 6))

        self.iteration_lbl = tk.Label(self.scrollable_info, text="", font=('Arial', 11, 'bold'), bg='#e3f2fd', fg='#1976d2')
        self.iteration_lbl.pack(pady=(2, 6))

        self.explored_lbl = tk.Label(self.scrollable_info, text="Explored Order:", font=('Arial', 11), bg='#e3f2fd', wraplength=260, justify='left')
        self.explored_lbl.pack(pady=(2, 6))

        self.frontier_lbl = tk.Label(self.scrollable_info, text="Frontier:", font=('Arial', 10), bg='#e3f2fd', fg='#333', wraplength=260, justify='left')
        self.frontier_lbl.pack(pady=(2, 6))

        self.cost_lbl = tk.Label(self.scrollable_info, text="", font=('Arial', 11, 'bold'), bg='#e3f2fd', fg='#d32f2f')
        self.cost_lbl.pack(pady=(2, 6))

        self.status_lbl = tk.Label(self.scrollable_info, text="Select algorithm and click Run Search",
                                   font=('Arial', 10), fg='#1565c0', bg='#e3f2fd', wraplength=260, justify='left')
        self.status_lbl.pack(pady=(2, 6))

        # Add iteration history for IDS
        self.iteration_history = tk.Text(self.scrollable_info, height=15, width=35, font=('Arial', 9), 
                                        bg='#ffffff', fg='#000000', wrap='word')
        self.iteration_history.pack(pady=(5, 5))
        self.iteration_history.pack_forget()  # Hidden by default

        self.on_algo_change()

    # --------------------------
    # Layout change
    # --------------------------
    def on_algo_change(self):
        algo = self.algo_var.get()
        # hide optional widgets
        self.depth_lbl.grid_remove()
        self.depth_spin.grid_remove()
        self.iter_lbl.grid_remove()
        self.iter_spin.grid_remove()
        self.iteration_history.pack_forget()

        if algo == 'DLS':
            self.depth_lbl.grid(row=0, column=6, padx=5)
            self.depth_spin.grid(row=0, column=7, padx=5)
        if algo == 'IDS':
            self.iter_lbl.grid(row=0, column=6, padx=5)
            self.iter_spin.grid(row=0, column=7, padx=5)
            self.iteration_history.pack(pady=(5, 5))

        if algo == 'UCS':
            self.graph = GRAPH_TREE2
            self.node_positions = copy.deepcopy(POS_TREE2)
            self.bg_img = self.tree2_img
        else:
            self.graph = GRAPH_TREE1
            self.node_positions = copy.deepcopy(POS_TREE1)
            self.bg_img = self.tree1_img

        self.draw_graph()

    def on_start_change(self):
        self.draw_graph()

    # --------------------------
    # Drawing helpers
    # --------------------------
    def arrow_endpoints(self, from_node, to_node):
        if from_node not in self.node_positions or to_node not in self.node_positions:
            return None
        x1, y1 = self.node_positions[from_node]
        x2, y2 = self.node_positions[to_node]
        dx, dy = x2 - x1, y2 - y1
        d = math.hypot(dx, dy) or 1.0
        pad = 6
        sx = x1 + (dx / d) * (self.node_radius + pad)
        sy = y1 + (dy / d) * (self.node_radius + pad)
        ex = x2 - (dx / d) * (self.node_radius + pad)
        ey = y2 - (dy / d) * (self.node_radius + pad)
        return sx, sy, ex, ey

    def draw_graph(self):
        self.canvas.delete('all')
        if getattr(self, 'bg_img', None):
            w = max(1, self.canvas.winfo_width())
            h = max(1, self.canvas.winfo_height())
            self.canvas.create_image(w/2, h/2, image=self.bg_img, tags=('bg',))

        drawn_edges = set()
        algo = self.algo_var.get()
        for node, neighbors in self.graph.items():
            if node not in self.node_positions:
                continue
            x1, y1 = self.node_positions[node]
            for child, cost in neighbors:
                if child not in self.node_positions:
                    continue
                edge_key = (node, child) if algo == 'UCS' else tuple(sorted((node, child)))
                if edge_key in drawn_edges:
                    continue
                drawn_edges.add(edge_key)
                x2, y2 = self.node_positions[child]
                self.canvas.create_line(x1, y1, x2, y2, width=2, fill='black', tags=("edge",))
                if algo == 'UCS':
                    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
                    dx, dy = x2 - x1, y2 - y1
                    norm = math.hypot(dx, dy) or 1.0
                    ox, oy = -dy / norm * 14, dx / norm * 14
                    self.canvas.create_text(mx + ox, my + oy, text=str(cost), font=('Arial', 10, 'bold'), fill='#d32f2f')

        # draw nodes
        self.node_items = {}
        start = self.start_var.get()
        for node, (x, y) in self.node_positions.items():
            color = 'yellow' if node == start else 'white'
            w = 3 if node == start else 2
            circle = self.canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                             x + self.node_radius, y + self.node_radius,
                                             fill=color, outline='black', width=w, tags=("node",))
            text = self.canvas.create_text(x, y, text=node, font=('Arial', 12, 'bold'), tags=("label",))
            self.node_items[node] = (circle, text)

    def color_node(self, node, color, tag="visited"):
        if node in self.node_items:
            circle, _ = self.node_items[node]
            self.canvas.itemconfig(circle, fill=color)
            self.canvas.addtag_withtag(tag, circle)

    def clear_dynamic(self):
        self.canvas.delete("dynamic")
        self.canvas.delete("arrow_current")
        self.canvas.delete("expand_arrow")
        self.canvas.delete("path_arrow")
        self.canvas.delete("limit")
        for node, _ in self.node_items.items():
            circle, _ = self.node_items.get(node, (None, None))
            if circle:
                base_color = 'yellow' if node == self.start_var.get() else 'white'
                self.canvas.itemconfig(circle, fill=base_color)
        self.explored_order = []
        self.expanded_edges = []

    # --------------------------
    # Draw animated dashed explored path (added)
    # --------------------------
    def draw_explored_dashes(self):
        """
        Draw dashed arrows for consecutive pairs in explored_order.
        Uses self.dash_offset to create the moving dashed effect.
        Tagged with 'explore_dash' and 'dynamic' so they are cleared with dynamic content.
        """
        # remove old dashed explore arrows
        self.canvas.delete("explore_dash")

        # draw dashed lines between consecutive explored nodes
        for i in range(len(self.explored_order) - 1):
            u = self.explored_order[i]
            v = self.explored_order[i + 1]
            pts = self.arrow_endpoints(u, v)
            if pts:
                sx, sy, ex, ey = pts
                # dash pattern approximates "- - - - ->"
                # dashoffset animates to produce moving dashes
                self.canvas.create_line(
                    sx, sy, ex, ey,
                    width=3,
                    dash=(10, 6),
                    dashoffset=self.dash_offset,
                    fill='#fb8c00',  # orange-ish to blend with expand animation
                    arrow=tk.LAST,
                    arrowshape=(12, 16, 6),
                    tags=("explore_dash", "dynamic")
                )

    # --------------------------
    # Algorithms with proper expand steps
    # --------------------------
    def bfs(self, start, goal):
        steps = []
        q = deque([start])
        parent = {start: None}
        visited = {start}
        steps.append({'type': 'frontier', 'frontier': list(q)})
        
        while q:
            node = q.popleft()
            steps.append({'type': 'visit', 'node': node, 'from': parent.get(node)})
            
            if node == goal:
                path = reconstruct_path(parent, node)
                steps.append({'type': 'found', 'path': path, 'cost': None})
                return steps
            
            # Expand children (left to right for BFS)
            for child, _ in self.graph.get(node, []):
                if child not in visited:
                    visited.add(child)
                    parent[child] = node
                    steps.append({'type': 'expand', 'from': node, 'to': child})
                    q.append(child)
            
            steps.append({'type': 'frontier', 'frontier': list(q)})
        
        steps.append({'type': 'notfound'})
        return steps

    def ucs(self, start, goal):
        steps = []
        costs = {start: 0}
        parent = {start: None}
        heap = [(0, start)]
        visited = set()
        steps.append({'type': 'frontier', 'frontier': [(start, 0)]})
        
        while heap:
            cost, node = heapq.heappop(heap)
            
            if node in visited:
                continue
            
            visited.add(node)
            steps.append({'type': 'visit', 'node': node, 'from': parent.get(node), 'cost': cost})
            
            if node == goal:
                path = reconstruct_path(parent, node)
                steps.append({'type': 'found', 'path': path, 'cost': cost})
                return steps
            
            for child, w in self.graph.get(node, []):
                newg = cost + w
                if child not in visited and newg < costs.get(child, float('inf')):
                    costs[child] = newg
                    parent[child] = node
                    steps.append({'type': 'expand', 'from': node, 'to': child, 'cost': newg})
                    heapq.heappush(heap, (newg, child))
            
            steps.append({'type': 'frontier', 'frontier': sorted([(n, c) for c, n in heap])})
        
        steps.append({'type': 'notfound'})
        return steps

    def dfs(self, start, goal):
        steps = []
        stack = [start]
        parent = {start: None}
        visited = {start}
        steps.append({'type': 'frontier', 'frontier': list(reversed(stack))})
        
        while stack:
            node = stack.pop()
            steps.append({'type': 'visit', 'node': node, 'from': parent.get(node)})
            
            if node == goal:
                path = reconstruct_path(parent, node)
                steps.append({'type': 'found', 'path': path, 'cost': None})
                return steps
            
            # Expand children in reverse order for DFS (to maintain left-to-right when popping)
            children = list(self.graph.get(node, []))
            for child, _ in reversed(children):
                if child not in visited:
                    visited.add(child)
                    parent[child] = node
                    steps.append({'type': 'expand', 'from': node, 'to': child})
                    stack.append(child)
            
            steps.append({'type': 'frontier', 'frontier': list(reversed(stack))})
        
        steps.append({'type': 'notfound'})
        return steps

    def dls_recursive(self, node, goal, depth, limit, parent, visited, steps):
        steps.append({'type': 'visit', 'node': node, 'from': parent.get(node), 'depth': depth})
        
        if node == goal:
            return True
        
        if depth >= limit:
            if any(True for _ in self.graph.get(node, [])):
                steps.append({'type': 'pruned', 'node': node, 'depth': depth})
            return False
        
        for child, _ in self.graph.get(node, []):
            if child not in visited:
                visited.add(child)
                parent[child] = node
                steps.append({'type': 'expand', 'from': node, 'to': child})
                if self.dls_recursive(child, goal, depth + 1, limit, parent, visited, steps):
                    return True
                visited.remove(child)
        
        return False

    def dls(self, start, goal, limit):
        steps = []
        parent = {start: None}
        visited = {start}
        steps.append({'type': 'limit', 'limit': limit})
        steps.append({'type': 'frontier', 'frontier': [start]})
        found = self.dls_recursive(start, goal, 0, limit, parent, visited, steps)
        
        if found:
            path = reconstruct_path(parent, goal)
            steps.append({'type': 'found', 'path': path, 'cost': None})
        else:
            steps.append({'type': 'notfound'})
        
        return steps

    def ids(self, start, goal, max_iter):
        all_steps = []
        
        for depth in range(max_iter + 1):
            all_steps.append({'type': 'iteration_start', 'depth': depth})
            all_steps.append({'type': 'limit', 'limit': depth})
            
            steps = []
            parent = {start: None}
            visited = {start}
            steps.append({'type': 'frontier', 'frontier': [start]})
            found = self.dls_recursive(start, goal, 0, depth, parent, visited, steps)
            all_steps.extend(steps)
            
            if found:
                path = reconstruct_path(parent, goal)
                all_steps.append({'type': 'found', 'path': path, 'cost': None})
                all_steps.append({'type': 'iteration_end', 'depth': depth, 'found': True})
                break
            else:
                all_steps.append({'type': 'iteration_end', 'depth': depth, 'found': False})
        
        return all_steps

    # --------------------------
    # Run / animate
    # --------------------------
    def run_search(self):
        if self.running:
            return
        self.reset(clear_ui=False)

        algo = self.algo_var.get()
        start = self.start_var.get()
        goal = self.goal_var.get()

        self.on_algo_change()

        self.status_lbl.config(text=f"Running {algo} from {start} to {goal}...", fg='#1565c0')
        
        if algo == 'BFS':
            self.steps = self.bfs(start, goal)
        elif algo == 'UCS':
            self.steps = self.ucs(start, goal)
        elif algo == 'DFS':
            self.steps = self.dfs(start, goal)
        elif algo == 'DLS':
            limit = int(self.depth_var.get())
            self.steps = self.dls(start, goal, limit)
        elif algo == 'IDS':
            max_iter = int(self.iter_var.get())
            self.steps = self.ids(start, goal, max_iter)
            self.iteration_history.delete('1.0', tk.END)
        else:
            self.steps = []

        self.step_index = 0
        self.explored_order = []
        self.expanded_edges = []
        self.running = True
        self.animate()

    def animate(self):
        if not self.running or self.step_index >= len(self.steps):
            self.running = False
            self.status_lbl.config(text="Animation finished.", fg='#2e7d32')
            # ensure final dashed arrows are drawn
            self.draw_explored_dashes()
            return

        step = self.steps[self.step_index]
        stype = step.get('type')

        if stype == 'iteration_start':
            self.clear_dynamic()
            depth = step['depth']
            self.iteration_lbl.config(text=f"=== Iteration {depth} (Depth Limit: {depth}) ===")
            self.iteration_history.insert(tk.END, f"\n{'='*40}\nIteration {depth} - Depth Limit: {depth}\n{'='*40}\n")
            self.iteration_history.see(tk.END)

        elif stype == 'iteration_end':
            depth = step['depth']
            found = step.get('found', False)
            if found:
                self.iteration_history.insert(tk.END, f"✓ Goal found at iteration {depth}!\n")
            else:
                self.iteration_history.insert(tk.END, f"✗ Goal not found at depth {depth}\n")
            self.iteration_history.see(tk.END)

        elif stype == 'limit':
            limit = step.get('limit', 0)
            self.draw_limit_line_by_level(limit)
            if self.algo_var.get() == 'DLS':
                self.status_lbl.config(text=f"DLS limit = {limit}", fg='#1565c0')

        elif stype == 'frontier':
            frontier = step.get('frontier', [])
            if frontier and isinstance(frontier[0], tuple):
                fld = ', '.join(f"{n}:{c}" for n, c in frontier)
            else:
                fld = ', '.join(str(x) for x in frontier)
            self.frontier_lbl.config(text=f"Frontier: {fld}")

        elif stype == 'visit':
            node = step['node']
            parent = step.get('from')
            cost = step.get('cost')
            depth = step.get('depth')

            self.canvas.delete("arrow_current")
            self.color_node(node, 'lightblue')
            
            if node not in self.explored_order:
                self.explored_order.append(node)
            
            self.step_lbl.config(text=f"Step: {len(self.explored_order)}")
            self.explored_lbl.config(text=f"Explored: {' → '.join(self.explored_order)}")
            
            if cost is not None:
                self.cost_lbl.config(text=f"Path Cost: {cost}")
            
            if self.algo_var.get() == 'IDS':
                depth_str = f" (depth {depth})" if depth is not None else ""
                self.iteration_history.insert(tk.END, f"Visiting: {node}{depth_str}\n")
                self.iteration_history.see(tk.END)

        elif stype == 'expand':
            frm = step.get('from')
            to = step.get('to')
            pts = self.arrow_endpoints(frm, to)
            
            if pts:
                sx, sy, ex, ey = pts
                self.canvas.create_line(sx, sy, ex, ey, fill='orange', width=3,
                                      arrow=tk.LAST, arrowshape=(12, 16, 6), 
                                      tags=("expand_arrow", "dynamic"))
                self.expanded_edges.append((frm, to))
            
            self.color_node(to, '#fff9c4')
            
            if self.algo_var.get() == 'IDS':
                self.iteration_history.insert(tk.END, f"  Expanding: {frm} → {to}\n")
                self.iteration_history.see(tk.END)

        elif stype == 'pruned':
            node = step.get('node')
            depth = step.get('depth')
            self.color_node(node, '#eceff1')
            if self.algo_var.get() == 'IDS':
                self.iteration_history.insert(tk.END, f"  Pruned: {node} (depth {depth})\n")
                self.iteration_history.see(tk.END)

        elif stype == 'found':
            path = step.get('path', [])
            cost = step.get('cost')
            
            # Color all explored nodes
            for n in self.explored_order:
                self.color_node(n, 'yellow')
            
            # Draw the exploration trail (expanded edges)
            for (u, v) in self.expanded_edges:
                pts = self.arrow_endpoints(u, v)
                if pts:
                    sx, sy, ex, ey = pts
                    self.canvas.create_line(sx, sy, ex, ey, fill='green', width=4,
                                          arrow=tk.LAST, arrowshape=(12, 16, 6), 
                                          tags=("path_arrow",))
            
            # Highlight path nodes
            if path:
                for n in path:
                    self.color_node(n, 'lime')
            
            path_str = ' → '.join(path) if path else ''
            
            if cost is not None:
                self.status_lbl.config(text=f"✓ Goal found! Path: {path_str}\nCost: {cost}", fg='#2e7d32')
                self.cost_lbl.config(text=f"Total Cost: {cost}")
            else:
                self.status_lbl.config(text=f"✓ Goal found! Path: {path_str}", fg='#2e7d32')
            
            if self.algo_var.get() == 'IDS':
                self.iteration_history.insert(tk.END, f"\n✓✓✓ GOAL FOUND! ✓✓✓\nPath: {path_str}\n")
                self.iteration_history.see(tk.END)

        elif stype == 'notfound':
            self.status_lbl.config(text="✗ Goal not found", fg='#d32f2f')
            if self.algo_var.get() == 'IDS':
                self.iteration_history.insert(tk.END, f"\n✗ Goal not found\n")
                self.iteration_history.see(tk.END)

        # Update dashed explored-arrow animation (moves dashes, and draws current explored edges)
        # increment dash offset to create motion and redraw dashed arrows between explored nodes
        self.dash_offset = (self.dash_offset + 6) % 100
        self.draw_explored_dashes()

        # Advance
        self.step_index += 1
        self.timer_id = self.root.after(self.speed_var.get(), self.animate)

    def draw_limit_line_by_level(self, level):
        self.canvas.delete("limit")
        if 0 <= level < len(LEVEL_Y):
            yline = LEVEL_Y[level] + 50
            self.canvas.create_line(60, yline, 850, yline, fill='red', width=3, 
                                   dash=(10, 5), tags=("limit",))
            self.canvas.create_text(70, yline - 10, text=f"Depth Limit: {level}", 
                                   font=('Arial', 10, 'bold'), fill='red', 
                                   anchor='w', tags=("limit",))

    def reset(self, clear_ui=True):
        if self.timer_id:
            try:
                self.root.after_cancel(self.timer_id)
            except Exception:
                pass
        
        self.running = False
        self.step_index = 0
        self.steps = []
        self.explored_order = []
        self.expanded_edges = []
        
        self.canvas.delete("arrow_current")
        self.canvas.delete("path_arrow")
        self.canvas.delete("expand_arrow")
        self.canvas.delete("limit")
        self.canvas.delete("dynamic")
        
        if clear_ui:
            self.draw_graph()
            self.step_lbl.config(text="Step: 0")
            self.iteration_lbl.config(text="")
            self.explored_lbl.config(text="Explored Order:")
            self.frontier_lbl.config(text="Frontier:")
            self.cost_lbl.config(text="")
            self.status_lbl.config(text="Select algorithm and click Run Search", fg='#1565c0')
            self.iteration_history.delete('1.0', tk.END)
        else:
            self.draw_graph()


# --------------------------
# Entry point
# --------------------------

def main():
    root = tk.Tk()

    root.geometry('1100x760')
    try:
        root.state('zoomed')
    except Exception:
        pass

    def _toggle_fullscreen(event=None):
        try:
            is_full = root.attributes('-fullscreen')
        except Exception:
            is_full = False
        root.attributes('-fullscreen', not is_full)

    root.bind('<F11>', _toggle_fullscreen)
    root.bind('<Escape>', lambda e: root.attributes('-fullscreen', False))

    app = SearchVisualizer(root)
    root.resizable(True, True)
    root.mainloop()


if __name__ == '__main__':
    main()
