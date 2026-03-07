import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.ndimage import convolve

@dataclass
class Params:
    # ── 地圖設定 ──────────────────────────────────────────────────────────────
    size: int = 60
    # 地圖邊長，產生 size × size 的方形網格

    empty_ratio: float = 0.3
    # 空屋比例 (0~1)。空屋是搬家的「緩衝空間」；太低會導致搬不動（Gridlock）

    group1_ratio: float = 0.7
    # 有住戶的格子中，Group1 所佔比例；剩餘為 Group2
    # 0.5 = 兩族人口相等；0.7 = 多數/少數族群情境

    # ── 隔離行為 ──────────────────────────────────────────────────────────────
    threshold_g1: float = 0.7
    # Group1 滿意門檻 (0~1)：鄰居中「同族比例」需達到此值才不想搬家
    # 0.5 = 一半同族即可；0.7 = 需要多數同族（會形成強烈隔離板塊）

    threshold_g2: float = 0.7
    # Group2 滿意門檻，定義同上

    friction_cost: float = 0.0
    # 搬家阻力 (0~1)：即使不滿意，也有 friction_cost 的機率忍耐不搬
    # 0.0 = 無阻力，稍有不滿立刻搬；0.8 = 只有 20% 機率會真的行動
    # 現實對應：房貸沉沒成本、學區綁定、搬家交易稅等

    # ── 市中心效應 ────────────────────────────────────────────────────────────
    cbd_gravity: float = 0.0
    # 市中心容忍度加成（用於 compute_unhappy）：
    # 距中心越近，有效門檻越低（有效門檻 = threshold - cbd_gravity）
    # cbd_gravity > threshold 時中心永遠滿意，自然形成雜居區
    # 現實對應：市中心的多元文化包容度較高

    cbd_gravity_g1: float = -1.0
    # Group1 搬家時的市中心偏好 (0~1)；-1 = 同 cbd_gravity
    # 窮人通常較難搶到市中心，設為低值或 0

    cbd_gravity_g2: float = -1.0
    # Group2 搬家時的市中心偏好 (0~1)；-1 = 同 cbd_gravity
    # 富人有更強的購買力，設為高值以優先搶佔市中心空地

    # ── 模擬控制 ──────────────────────────────────────────────────────────────
    max_steps: int = 200
    # 最多執行幾個時間步後停止

    neighborhood: str = "von_neumann"
    # 鄰居定義："moore"（8 宮格）或 "von_neumann"（上下左右 4 格）
    # moore 產生較平滑的板塊邊界；von_neumann 邊界較銳利

    def __post_init__(self):
        if self.cbd_gravity_g1 < 0:
            self.cbd_gravity_g1 = self.cbd_gravity
        if self.cbd_gravity_g2 < 0:
            self.cbd_gravity_g2 = self.cbd_gravity

# cell state: 0=empty, 1=group1, 2=group2
def init_grid(p: Params, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = p.size * p.size
    n_empty = int(n * p.empty_ratio)
    n_occ = n - n_empty
    n_g1 = int(n_occ * p.group1_ratio)
    n_g2 = n_occ - n_g1

    arr = np.array([0]*n_empty + [1]*n_g1 + [2]*n_g2, dtype=np.int8)
    rng.shuffle(arr)
    return arr.reshape((p.size, p.size))

def compute_unhappy(grid: np.ndarray, p: Params) -> np.ndarray:
    """回傳與 grid 同形的 bool 陣列，True 代表該格居民不滿意（想搬家）。"""
    H, W = grid.shape
    center_r = (p.size - 1) / 2.0
    center_c = (p.size - 1) / 2.0

    # 用 convolution 一次計算所有格子的鄰居統計，避免逐格 Python 迴圈
    if p.neighborhood == "von_neumann":
        kernel = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]], dtype=float)
    else:
        kernel = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], dtype=float)

    same1        = convolve((grid == 1).astype(float), kernel, mode='constant', cval=0)
    same2        = convolve((grid == 2).astype(float), kernel, mode='constant', cval=0)
    occ_neighbors = convolve((grid != 0).astype(float), kernel, mode='constant', cval=0)

    # 市中心容忍度加成：距離中心越近，有效門檻越低（最多降低 cbd_gravity）
    # 當 cbd_gravity >= threshold 時，中心格子永遠滿意，自然形成雜居區
    rows = np.arange(H)[:, None]
    cols = np.arange(W)[None, :]
    dist = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
    location_bonus = np.clip(1.0 - (dist / (p.size * 0.2)), 0, 1) * p.cbd_gravity

    # 無鄰居時視為滿意（ratio = 1.0）
    with np.errstate(invalid='ignore'):
        ratio1 = np.where(occ_neighbors > 0, same1 / occ_neighbors, 1.0)
        ratio2 = np.where(occ_neighbors > 0, same2 / occ_neighbors, 1.0)

    unhappy1 = (grid == 1) & (ratio1 < (p.threshold_g1 - location_bonus))
    unhappy2 = (grid == 2) & (ratio2 < (p.threshold_g2 - location_bonus))

    return unhappy1 | unhappy2

def _do_moves(grid: np.ndarray, movers: np.ndarray, empty_pos: np.ndarray,
              gravity: float, rng: np.random.Generator):
    """
    對一個群體執行搬家動作。
    回傳（更新後的空位列表, 實際搬家人數）。
    搬走者釋放的舊位置會加回空位列表，供下一個群體使用。
    """
    if len(movers) == 0 or len(empty_pos) == 0:
        return empty_pos, 0

    if gravity > 0:
        H, W = grid.shape
        center_r = (H - 1) / 2.0
        center_c = (W - 1) / 2.0
        distances      = np.sqrt((empty_pos[:, 0] - center_r)**2 + (empty_pos[:, 1] - center_c)**2)
        attractiveness = np.clip(1.0 - (distances / (H / 1.414)), 0, 1)
        weights        = gravity * attractiveness + (1.0 - gravity) * rng.random(len(empty_pos))
        empty_pos = empty_pos[np.argsort(weights)[::-1]]
    else:
        rng.shuffle(empty_pos)

    n = min(len(movers), len(empty_pos))
    freed = []
    for i in range(n):
        ur, uc = movers[i]
        er, ec = empty_pos[i]
        grid[er, ec] = grid[ur, uc]
        grid[ur, uc] = 0
        freed.append([ur, uc])

    # 剩餘空位 = 未被選用的空位 + 搬走後釋放的舊位置
    parts = [x for x in [empty_pos[n:] if n < len(empty_pos) else None,
                         np.array(freed, dtype=int) if freed else None] if x is not None]
    remaining = np.vstack(parts) if parts else np.empty((0, 2), dtype=int)
    return remaining, n


def step(grid: np.ndarray, p: Params, rng: np.random.Generator):
    unhappy = compute_unhappy(grid, p)
    unhappy_pos = np.argwhere(unhappy)
    empty_pos = np.argwhere(grid == 0)

    total_unhappy = unhappy_pos.shape[0]

    if total_unhappy == 0 or empty_pos.size == 0:
        return grid, 0, total_unhappy

    # 搬家阻力：不滿意的居民仍有 friction_cost 機率選擇忍耐不動
    # 模擬房貸綁定、學區成本、交易稅等現實摩擦
    will_move_mask = rng.random(total_unhappy) < (1.0 - p.friction_cost)
    actual_movers = unhappy_pos[will_move_mask]

    if len(actual_movers) == 0:
        return grid, 0, total_unhappy  # 所有人都被阻力卡住（Gridlock）

    rng.shuffle(actual_movers)

    # 分群搶地：Group2（富人）優先選市中心空地，Group1（窮人）再從剩餘中選
    # 模擬富人購買力較強，能優先搶到精華地段
    g2_mask   = np.array([grid[r, c] == 2 for r, c in actual_movers])
    g2_movers = actual_movers[g2_mask]
    g1_movers = actual_movers[~g2_mask]

    empty_pos, n2 = _do_moves(grid, g2_movers, empty_pos, p.cbd_gravity_g2, rng)
    empty_pos, n1 = _do_moves(grid, g1_movers, empty_pos, p.cbd_gravity_g1, rng)

    return grid, n1 + n2, total_unhappy

def run(p: Params, seed: int = 7, animate: bool = True):
    rng = np.random.default_rng(seed)
    grid = init_grid(p, seed=seed)

    from matplotlib.colors import ListedColormap
    # 格子顏色：0=空屋（黑）、1=Group1（藍綠）、2=Group2（金）
    cmap = ListedColormap(["#1e1e1e", "#20B2AA", "#FFD700"])
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")

    # CBD 影響範圍：虛線紅圈 + 中心十字
    if p.cbd_gravity > 0:
        center = p.size / 2.0
        circle = plt.Circle((center, center), p.size*0.2, color='red',
                             fill=False, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        ax.plot(center, center, 'r+', markersize=12)

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
            f"Step: {stats['step']} | Unhappy: {unhappy_n} | Moved: {moves} | Stuck: {unhappy_n - moves}"
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
        plt.figure(figsize=(7,7))
        plt.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        if p.cbd_gravity > 0:
            center = p.size / 2.0
            plt.plot(center, center, 'r+', markersize=12)
            circle = plt.Circle((center, center), p.size*0.2, color='red',
                                 fill=False, linestyle='--', linewidth=2, alpha=0.5)
            plt.gca().add_patch(circle)
        plt.title(f"Final | th1={p.threshold_g1}, th2={p.threshold_g2}")
        plt.axis("off")
        plt.show()
        return grid

if __name__ == "__main__":
    p = Params(
        size=60,
        empty_ratio=0.2,      # 20% 空屋，提供足夠搬家緩衝空間
        group1_ratio=0.5,     # 兩族人口各佔一半
        threshold_g1=0.65,    # 高門檻 → 外部強烈隔離（8鄰需 ≥6 個同類）
        threshold_g2=0.65,
        friction_cost=0.0,    # 無阻力：不滿意立刻行動
        cbd_gravity=0.8,      # 0.8 > threshold 0.65 → 市中心永遠滿意，自然雜居
        max_steps=500,
        neighborhood="moore", # 8 宮格：板塊邊界平滑
    )
    run(p, seed=7, animate=True)
