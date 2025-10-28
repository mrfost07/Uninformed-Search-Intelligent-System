# 🔍 Graph Search Algorithm Visualizer

An interactive visualization tool for understanding classical graph search algorithms. Watch BFS, DFS, UCS, DLS, and IDS algorithms come to life with animated nodes, edges, and exploration paths.

---

## ⚡ Quick Start

### Installation
```bash
git clone https://github.com/mrfost07/Uninformed-Search-Intelligent-System.git
cd Uninformed-Search-Intelligent-System
```

### Requirements
- Python 3.8+
- tkinter (included with Python, or `sudo apt-get install python3-tk` on Linux)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run
```bash
python3 main.py
```

---

## 🧭 Algorithms

### BFS — Breadth-First Search
**Strategy:** Explore nodes level by level using a queue (FIFO)  
**Best for:** Unweighted graphs, finding shortest path by steps  
**Key trait:** Expands all nodes at depth N before depth N+1  
**Optimality:** ✅ Finds shortest path

![BFS](https://github.com/mrfost07/Uninformed-Search-Intelligent-System/blob/main/screenshots/bfs.png?raw=true)

---

### DFS — Depth-First Search
**Strategy:** Dive deep into branches using a stack (LIFO), then backtrack  
**Best for:** Checking connectivity, topological sorting  
**Key trait:** Explores one branch fully before switching  
**Optimality:** ❌ Not guaranteed shortest path

![DFS](https://github.com/mrfost07/Uninformed-Search-Intelligent-System/blob/main/screenshots/dfs.png?raw=true)

---

### UCS — Uniform Cost Search
**Strategy:** Always expand the lowest-cost node using a priority queue  
**Best for:** Weighted graphs, finding lowest-cost path  
**Key trait:** Considers cumulative edge weights (like Dijkstra)  
**Optimality:** ✅ Finds optimal-cost path

![UCS](https://github.com/mrfost07/Uninformed-Search-Intelligent-System/blob/main/screenshots/ucs.png?raw=true)

---

### DLS — Depth-Limited Search
**Strategy:** DFS with a maximum depth limit to prevent infinite loops  
**Best for:** Bounded search in unknown depths  
**Key trait:** Stops exploring beyond set depth limit  
**Optimality:** ❌ Like DFS, but depth-bounded

![DLS](https://github.com/mrfost07/Uninformed-Search-Intelligent-System/blob/main/screenshots/dls.png?raw=true)

---

### IDS — Iterative Deepening Search
**Strategy:** Repeatedly run DLS with increasing depth limits (1, 2, 3, ...)  
**Best for:** Very large graphs, memory-efficient optimal search  
**Key trait:** Combines BFS optimality with DFS memory efficiency  
**Optimality:** ✅ Finds shortest path (like BFS)

![IDS](https://github.com/mrfost07/Uninformed-Search-Intelligent-System/blob/main/screenshots/ids.png?raw=true)

---

## 🎮 Controls

| Control | Purpose |
|---------|---------|
| **Algorithm** | Select BFS, DFS, UCS, DLS, or IDS |
| **Start / Goal** | Choose entry and target nodes |
| **Speed (ms)** | Animation step delay (50–1000ms) |
| **Depth Limit** | For DLS: max exploration depth |
| **Max Iter** | For IDS: number of iterations |
| **Run Search** | Start visualization |
| **Reset** | Clear and stop |

### Keyboard Shortcuts
- `Space` — pause/resume
- `R` — reset
- `F11` — fullscreen

---

## 🎨 Visualization Legend

| Element | Meaning |
|---------|---------|
| 🟢 Green node | Goal found |
| 🟡 Yellow node | Start node |
| ⚪ White node | Unvisited |
| 🔵 Light blue node | Visited |
| 🟠 Orange dashed arrow | Explored path progression |
| ➜ Solid arrow | Node expansion |
| 🔴 Red dashed line (DLS) | Depth limit boundary |

---

## 📚 Learn More

For installation issues, see [Troubleshooting](#troubleshooting) below.

### Troubleshooting

**tkinter import error?**  
- Ubuntu/Debian: `sudo apt-get install python3-tk`
- Fedora: `sudo dnf install python3-tkinter`
- macOS: Reinstall Python via `brew`
- Windows: Reinstall Python and select tkinter during setup

**Images not loading?**  
The app continues without background images—this is normal if image files are missing.

---

## 📖 References

- **AI Textbook**: Russell & Norvig, *Artificial Intelligence: A Modern Approach*
- **Python Docs**: [tkinter official docs](https://docs.python.org/3/library/tkinter.html)

---

## 📝 License

MIT License — feel free to use and modify!
