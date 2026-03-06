import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass

@dataclass
class Params:
    size: int = 100             # 地圖邊長 (size x size)
    empty_ratio: float = 0.3  # 空屋比例
    group_ratio: float = 0.7   # 非空屋中，group1 的比例；group2 = 1 - group_ratio
    threshold: float = 0.7    # 滿意門檻：鄰居中同族比例 >= threshold
    max_steps: int = 200       # 最多迭代步數
    neighborhood: str = "von_neumann"  # "moore"(8鄰) 或 "von_neumann"(4鄰)

# cell state: 0=empty, 1=group1, 2=group2
def init_grid(p: Params, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = p.size * p.size
    n_empty = int(n * p.empty_ratio)
    n_occ = n - n_empty
    n_g1 = int(n_occ * p.group_ratio)
    n_g2 = n_occ - n_g1

    arr = np.array([0]*n_empty + [1]*n_g1 + [2]*n_g2, dtype=np.int8)
    rng.shuffle(arr)
    return arr.reshape((p.size, p.size))

def neighbor_offsets(p: Params):
    if p.neighborhood == "von_neumann":
        return [(-1,0),(1,0),(0,-1),(0,1)]
    # moore
    return [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

def compute_unhappy(grid: np.ndarray, p: Params) -> np.ndarray:
    """Return boolean mask of unhappy occupied cells."""
    offsets = neighbor_offsets(p)
    H, W = grid.shape
    unhappy = np.zeros((H, W), dtype=bool)

    # For each cell that is not empty, compute ratio of same-type among occupied neighbors
    for r in range(H):
        for c in range(W):
            t = grid[r, c]
            if t == 0:
                continue

            same = 0
            occ = 0
            for dr, dc in offsets:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    nt = grid[rr, cc]
                    if nt != 0:
                        occ += 1
                        if nt == t:
                            same += 1

            # 若周圍沒有任何住戶：視為滿意（你也可以改成不滿意）
            if occ == 0:
                unhappy[r, c] = False
            else:
                unhappy[r, c] = (same / occ) < p.threshold

    return unhappy

def step(grid: np.ndarray, p: Params, rng: np.random.Generator):
    unhappy = compute_unhappy(grid, p)
    unhappy_pos = np.argwhere(unhappy)
    empty_pos = np.argwhere(grid == 0)

    if unhappy_pos.size == 0 or empty_pos.size == 0:
        return grid, 0, unhappy_pos.shape[0]

    # 讓不滿意的人搬家：搬到隨機空位
    rng.shuffle(unhappy_pos)
    rng.shuffle(empty_pos)

    moves = min(len(unhappy_pos), len(empty_pos))
    for i in range(moves):
        ur, uc = unhappy_pos[i]
        er, ec = empty_pos[i]
        grid[er, ec] = grid[ur, uc]
        grid[ur, uc] = 0

    return grid, moves, unhappy_pos.shape[0]

def run(p: Params, seed: int = 7, animate: bool = True):
    rng = np.random.default_rng(seed)
    grid = init_grid(p, seed=seed)

    # plot setup
    cmap = plt.cm.get_cmap("viridis", 3)  # 0,1,2 三個值
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    stats = {"step": 0, "moves": 0, "unhappy": 0}

    def update(frame):
        nonlocal grid
        grid, moves, unhappy_n = step(grid, p, rng)
        stats["step"] += 1
        stats["moves"] = moves
        stats["unhappy"] = unhappy_n

        im.set_data(grid)
        title.set_text(
            f"Schelling Segregation | step={stats['step']} | unhappy={unhappy_n} | moves={moves} | threshold={p.threshold}"
        )
        return [im, title]

    if animate:
        ani = animation.FuncAnimation(
            fig, update, frames=p.max_steps, interval=80, blit=False, repeat=False
        )
        plt.show()
        return ani
    else:
        for _ in range(p.max_steps):
            grid, moves, unhappy_n = step(grid, p, rng)
            if unhappy_n == 0 or moves == 0:
                break
        plt.figure(figsize=(6,6))
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        plt.title(f"Final | threshold={p.threshold}")
        plt.axis("off")
        plt.show()
        return grid

if __name__ == "__main__":
    p = Params(
        size=60,
        empty_ratio=0.2,
        group_ratio=0.5,
        threshold=0.55,      # 0.3~0.5 會很有感
        max_steps=300,
        neighborhood="von_neumann"#"moore" #"von_neumann"
    )
    run(p, seed=7, animate=True)
