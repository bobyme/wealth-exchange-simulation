import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from dataclasses import dataclass
from scipy.ndimage import convolve

# 設定中文字體（macOS）
plt.rcParams['font.family'] = ['Arial Unicode MS', 'PingFang SC', 'Heiti TC', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False  # 修正負號顯示

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

    # ── 價格機制（price_enabled=True 時啟用）────────────────────────────────────
    price_enabled: bool = False
    # 啟用後新增房價陣列：富人入住推高地價，窮人若負擔不起會被強制驅逐（仕紳化）

    price_floor: float = 0.5
    # 地價下限（即使完全空置也不低於此值）

    price_ceiling: float = 3.0
    # 地價上限（富人大量聚集也不超過）

    price_cbd_premium: float = 1.0
    # 初始 CBD 溢價：市中心比邊緣貴多少（加在 price_floor 上）

    price_appreciation_rate: float = 0.05
    # 富人入住每步的直接漲幅；鄰近富人的空間外溢效果為此值的 50%

    price_decay_rate: float = 0.02
    # 空置或窮人入住每步的跌幅

    income_limit_g1: float = 1.5
    # Group1（窮人）能負擔的最高房價；超過即被強制搬離（不論鄰居是否滿意）

    income_limit_g2: float = 99.0
    # Group2（富人）的負擔上限（遠超 price_ceiling，幾乎不受限）

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

def init_price(p: Params) -> np.ndarray:
    """初始化地價陣列：市中心起始較貴，邊緣較便宜。"""
    center_r = (p.size - 1) / 2.0
    center_c = (p.size - 1) / 2.0
    rows = np.arange(p.size)[:, None]
    cols = np.arange(p.size)[None, :]
    dist = np.sqrt((rows - center_r)**2 + (cols - center_c)**2)
    cbd_gradient = np.clip(1.0 - dist / (p.size / 1.414), 0, 1)
    return p.price_floor + p.price_cbd_premium * cbd_gradient

def update_price(grid: np.ndarray, price: np.ndarray, p: Params) -> None:
    """就地更新地價（in-place）。富人入住漲，空置/窮人跌，鄰近富人有外溢效果。"""
    delta = np.where(grid == 2,  p.price_appreciation_rate,
            np.where(grid == 1, -p.price_decay_rate * 0.3,
                                -p.price_decay_rate))
    # 鄰近富人的空間外溢：帶動周圍地價
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float) / 8
    neighbor_rich = convolve((grid == 2).astype(float), kernel, mode='wrap')
    spillover = neighbor_rich * p.price_appreciation_rate * 0.5
    price[:] = np.clip(price + delta + spillover, p.price_floor, p.price_ceiling)

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

    same1         = convolve((grid == 1).astype(float), kernel, mode='wrap')
    same2         = convolve((grid == 2).astype(float), kernel, mode='wrap')
    occ_neighbors = convolve((grid != 0).astype(float), kernel, mode='wrap')

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
              gravity: float, rng: np.random.Generator,
              price: np.ndarray = None, income_limit: float = None):
    """
    對一個群體執行搬家動作。
    price / income_limit 不為 None 時，只考慮負擔得起的空地。
    回傳（更新後的空位列表, 實際搬家人數）。
    搬走者釋放的舊位置會加回空位列表，供下一個群體使用。
    """
    if len(movers) == 0 or len(empty_pos) == 0:
        return empty_pos, 0

    # 負擔能力過濾：只保留 price <= income_limit 的空地
    if price is not None and income_limit is not None:
        affordable = price[empty_pos[:, 0], empty_pos[:, 1]] <= income_limit
        empty_pos = empty_pos[affordable]
        if len(empty_pos) == 0:
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


def step(grid: np.ndarray, p: Params, rng: np.random.Generator,
         price: np.ndarray = None):
    # ── 1. 更新地價（若啟用）────────────────────────────────────────────────
    if p.price_enabled and price is not None:
        update_price(grid, price, p)

    # ── 2. 計算需要搬家的人 ─────────────────────────────────────────────────
    unhappy = compute_unhappy(grid, p)

    # 價格驅逐：Group1 若負擔不起當前格子，強制搬離（不論鄰居是否滿意）
    priced_out = np.zeros(grid.shape, dtype=bool)
    if p.price_enabled and price is not None:
        priced_out = (grid == 1) & (price > p.income_limit_g1)

    must_move   = unhappy | priced_out
    must_move_pos = np.argwhere(must_move)
    empty_pos   = np.argwhere(grid == 0)

    total_unhappy = int(unhappy.sum())  # 回報原始不滿意數（不含被驅逐者）

    if must_move_pos.shape[0] == 0 or empty_pos.size == 0:
        return grid, 0, total_unhappy

    # ── 3. 搬家阻力（被驅逐者無條件搬，其他人有機率忍耐）──────────────────
    is_priced_out = priced_out[must_move_pos[:, 0], must_move_pos[:, 1]]
    will_move_mask = is_priced_out | (
        rng.random(must_move_pos.shape[0]) < (1.0 - p.friction_cost)
    )
    actual_movers = must_move_pos[will_move_mask]

    if len(actual_movers) == 0:
        return grid, 0, total_unhappy  # 所有人都被阻力卡住（Gridlock）

    rng.shuffle(actual_movers)

    # ── 4. 分群搶地：富人優先，窮人只能選負擔得起的剩餘空地 ────────────────
    g2_mask   = np.array([grid[r, c] == 2 for r, c in actual_movers])
    g2_movers = actual_movers[g2_mask]
    g1_movers = actual_movers[~g2_mask]

    price_g1 = price if (p.price_enabled and price is not None) else None
    price_g2 = price if (p.price_enabled and price is not None) else None

    empty_pos, n2 = _do_moves(grid, g2_movers, empty_pos, p.cbd_gravity_g2, rng,
                               price=price_g2, income_limit=p.income_limit_g2)
    empty_pos, n1 = _do_moves(grid, g1_movers, empty_pos, p.cbd_gravity_g1, rng,
                               price=price_g1, income_limit=p.income_limit_g1)

    return grid, n1 + n2, total_unhappy

def _setup_axes(p: Params):
    """建立圖形與座標軸，若啟用價格機制則左右並排。"""
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#1e1e1e", "#20B2AA", "#FFD700"])
    ncols = 2 if p.price_enabled else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    ax = axes[0] if p.price_enabled else axes
    return fig, ax, axes, cmap

def _draw_cbd(ax, p: Params):
    if p.cbd_gravity > 0:
        center = p.size / 2.0
        circle = plt.Circle((center, center), p.size * 0.2, color='red',
                             fill=False, linestyle='--', linewidth=2, alpha=0.5)
        ax.add_patch(circle)
        ax.plot(center, center, 'r+', markersize=12)

def run(p: Params, seed: int = 7, animate: bool = True):
    rng  = np.random.default_rng(seed)
    grid = init_grid(p, seed=seed)
    price = init_price(p) if p.price_enabled else None

    fig, ax, axes, cmap = _setup_axes(p)
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
    _draw_cbd(ax, p)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ax.set_title("")

    # 地價熱圖（右側，僅 price_enabled 時存在）
    im_price = None
    if p.price_enabled:
        ax_p = axes[1]
        im_price = ax_p.imshow(price, cmap='hot',
                                vmin=p.price_floor, vmax=p.price_ceiling,
                                interpolation="nearest")
        plt.colorbar(im_price, ax=ax_p, fraction=0.046, pad=0.04, label='地價')
        ax_p.set_xticks([])
        ax_p.set_yticks([])
        ax_p.set_title("地價熱圖")

    stats = {"step": 0, "moves": 0, "unhappy": 0}

    def update(frame):
        nonlocal grid
        grid, moves, unhappy_n = step(grid, p, rng, price)
        stats["step"] += 1

        im.set_data(grid)
        if im_price is not None:
            im_price.set_data(price)  # price 已 in-place 更新
        title.set_text(
            f"Step: {stats['step']} | Unhappy: {unhappy_n} | Moved: {moves}"
        )
        return [im, title] + ([im_price] if im_price is not None else [])

    if animate:
        ani = animation.FuncAnimation(
            fig, update, frames=p.max_steps, interval=80, blit=False, repeat=False
        )
        plt.tight_layout()
        plt.show()
        return ani
    else:
        for _ in range(p.max_steps):
            grid, moves, unhappy_n = step(grid, p, rng, price)
            if unhappy_n == 0 or moves == 0:
                break
        ncols = 2 if p.price_enabled else 1
        fig2, axes2 = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
        ax2 = axes2[0] if p.price_enabled else axes2
        ax2.imshow(grid, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        _draw_cbd(ax2, p)
        ax2.set_title(f"Final | th1={p.threshold_g1}, th2={p.threshold_g2}")
        ax2.axis("off")
        if p.price_enabled:
            ax2_p = axes2[1]
            im2_p = ax2_p.imshow(price, cmap='hot',
                                  vmin=p.price_floor, vmax=p.price_ceiling,
                                  interpolation="nearest")
            plt.colorbar(im2_p, ax=ax2_p, fraction=0.046, pad=0.04, label='地價')
            ax2_p.set_title("最終地價分布")
            ax2_p.axis("off")
        plt.tight_layout()
        plt.show()
        return grid

if __name__ == "__main__":
    p = Params(
        size=60,
        empty_ratio=0.05,          # 20% 空屋，提供足夠搬家緩衝空間
        group1_ratio=0.5,         # 兩族人口各佔一半
        threshold_g1=0.65,        # 高門檻 → 外部強烈隔離（8鄰需 ≥6 個同類）
        threshold_g2=0.65,
        friction_cost=0.0,        # 無阻力：不滿意立刻行動
        cbd_gravity=0.8,          # 0.8 > threshold 0.65 → 市中心永遠滿意，自然雜居
        cbd_gravity_g1=0.2,       # 窮人較難搶市中心
        cbd_gravity_g2=0.95,      # 富人強力搶市中心
        max_steps=500,
        neighborhood="moore",     # 8 宮格：板塊邊界平滑
        price_enabled=True,       # 啟用仕紳化機制
        price_cbd_premium=1.0,    # 市中心初始溢價
        income_limit_g1=1.5,      # 窮人負擔上限（超過即被驅逐）
    )
    run(p, seed=7, animate=True)
