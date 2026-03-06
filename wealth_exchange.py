import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import CheckButtons


def gini(x: np.ndarray) -> float:
    """計算基尼係數，並處理非有限數值。"""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0 or np.all(x <= 0):
        return 0.0
    x_sorted = np.sort(x)
    n = x_sorted.size
    cum = np.cumsum(x_sorted)
    denom = cum[-1]
    if denom == 0:
        return 0.0
    return (n + 1 - 2 * np.sum(cum) / denom) / n


def simulate_exchange(
    n_agents: int = 1000,
    steps: int = 200_000,
    seed: int = 7,
    saving: float = 0.0,
    saving_slope: float = 0.0,
    saving_min: float = 0.0,
    saving_max: float = 0.95,
    sample_every: int = 100,
    metrics: list[str] | None = None,
    tax_rate_labor: float = 0.0,
    tax_rate_capital: float = 0.0,
    high_skill_ratio: float = 0.0,
    labor_income: float = 0.0,
    labor_vol: float = 0.0,
    capital_return: float = 0.0,
    high_skill_bonus: float = 1.0,
):
    rng = np.random.default_rng(seed)
    w = np.ones(n_agents, dtype=float)
    high_skill = rng.random(n_agents) < high_skill_ratio

    all_possible_metrics = ["gini", "mean", "median", "p90_p10", "top10_share"]
    series = {m: [] for m in all_possible_metrics}
    times = []

    for t in range(steps):
        # 1. 隨機交換 (Dimensionless transaction)
        i, j = rng.integers(0, n_agents, size=2)
        if i != j:
            wi, wj = w[i], w[j]
            mean_w = np.mean(w)
            si = np.clip(saving + saving_slope * (wi / (mean_w or 1.0) - 1.0), saving_min, saving_max)
            sj = np.clip(saving + saving_slope * (wj / (mean_w or 1.0) - 1.0), saving_min, saving_max)
            saved_i, saved_j = si * wi, sj * wj
            traded = (wi - saved_i) + (wj - saved_j)
            eps = rng.random()
            w[i] = saved_i + eps * traded
            w[j] = saved_j + (1.0 - eps) * traded

        # 2. 勞動收入與稅收
        total_labor_tax = 0.0
        if labor_income > 0.0:
            wages = np.full(n_agents, labor_income, dtype=float)
            if labor_vol > 0.0:
                wages += rng.normal(0.0, labor_vol, size=n_agents)
                wages = np.clip(wages, 0.0, None)
            if high_skill_ratio > 0.0:
                wages[high_skill] *= high_skill_bonus
            
            if tax_rate_labor > 0.0:
                tax = wages * tax_rate_labor
                wages -= tax
                total_labor_tax = np.sum(tax)
            w += wages

        # 3. 資本回報與稅收
        total_capital_tax = 0.0
        if capital_return > 0.0:
            w *= (1.0 + capital_return)
        
        if tax_rate_capital > 0.0:
            tax = w * tax_rate_capital
            w -= tax
            total_capital_tax = np.sum(tax)

        # 4. 再分配 (UBI)
        total_tax = total_labor_tax + total_capital_tax
        if total_tax > 0.0:
            w += total_tax / n_agents

        if not np.all(np.isfinite(w)):
            w = np.nan_to_num(w, nan=0.0, posinf=1e12, neginf=0.0)

        if sample_every > 0 and (t + 1) % sample_every == 0:
            times.append(t + 1)
            w_sorted = np.sort(w)
            sum_w = np.sum(w_sorted) or 1.0
            series["gini"].append(gini(w_sorted))
            series["mean"].append(float(np.mean(w_sorted)))
            series["median"].append(float(np.median(w_sorted)))
            p90, p10 = np.percentile(w_sorted, [90, 10])
            series["p90_p10"].append(float(p90 / max(p10, 1e-9)))
            top10 = int(0.1 * n_agents)
            series["top10_share"].append(float(np.sum(w_sorted[-top10:]) / sum_w))

    return w, times, series


def run_animation(
    n_agents: int, steps: int, saving: float, seed: int,
    saving_slope: float, saving_min: float, saving_max: float,
    sample_every: int, metrics: list[str], tax_rate_labor: float,
    tax_rate_capital: float, high_skill_ratio: float,
    labor_income: float, labor_vol: float, capital_return: float,
    high_skill_bonus: float
):
    rng = np.random.default_rng(seed)
    w = np.ones(n_agents, dtype=float)
    high_skill = rng.random(n_agents) < high_skill_ratio

    all_metrics = ["gini", "mean", "median", "p90_p10", "top10_share"]
    series = {m: [] for m in all_metrics}
    times = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(right=0.85)

    lines = {}
    for m in all_metrics:
        line, = axes[1].plot([], [], label=m)
        line.set_visible(m in metrics)
        lines[m] = line
    
    axes[1].set_title("Metrics Over Time")
    axes[1].set_xlabel("Steps")
    axes[1].set_xlim(0, steps)
    axes[1].legend(loc="upper left", fontsize='small')

    check_ax = fig.add_axes([0.88, 0.3, 0.1, 0.4])
    checks = CheckButtons(check_ax, all_metrics, [m in metrics for m in all_metrics])
    fig._checks_reference = checks

    def _on_toggle(label):
        lines[label].set_visible(not lines[label].get_visible())
        axes[1].legend([l for l in lines.values() if l.get_visible()],
                       [m for m, l in lines.items() if l.get_visible()],
                       loc="upper left", fontsize='small')
        fig.canvas.draw_idle()
    checks.on_clicked(_on_toggle)

    def update(frame):
        nonlocal w
        for _ in range(sample_every):
            i, j = rng.integers(0, n_agents, size=2)
            if i != j:
                wi, wj = w[i], w[j]
                mean_w = np.mean(w)
                si = np.clip(saving + saving_slope * (wi/(mean_w or 1.0) - 1.0), saving_min, saving_max)
                sj = np.clip(saving + saving_slope * (wj/(mean_w or 1.0) - 1.0), saving_min, saving_max)
                traded = (wi - si*wi) + (wj - sj*wj)
                eps = rng.random()
                w[i], w[j] = si*wi + eps*traded, sj*wj + (1.0-eps)*traded
            
            if labor_income > 0:
                wages = np.full(n_agents, labor_income) + rng.normal(0, labor_vol, n_agents)
                wages = np.clip(wages, 0, None)
                wages[high_skill] *= high_skill_bonus
                tax_l = wages * tax_rate_labor
                w += (wages - tax_l) + np.sum(tax_l)/n_agents
            
            if capital_return > 0:
                w *= (1.0 + capital_return)
                tax_c = w * tax_rate_capital
                w = (w - tax_c) + np.sum(tax_c)/n_agents
            
            if not np.all(np.isfinite(w)):
                w = np.nan_to_num(w, posinf=1e12, nan=0)

        t = (frame + 1) * sample_every
        times.append(t)
        w_sorted = np.sort(w)
        series["gini"].append(gini(w_sorted))
        series["mean"].append(float(np.mean(w_sorted)))
        series["median"].append(float(np.median(w_sorted)))
        p90, p10 = np.percentile(w_sorted, [90, 10])
        series["p90_p10"].append(float(p90 / max(p10, 1e-9)))
        series["top10_share"].append(float(np.sum(w_sorted[-int(0.1*n_agents):]) / (np.sum(w_sorted) or 1.0)))

        axes[0].cla()
        lo, hi = np.percentile(w, [0, 99])
        if hi <= lo: hi = lo + 1.0
        axes[0].hist(np.clip(w, lo, hi), bins=40, density=True, color="#2a6f97", edgecolor="white")
        axes[0].set_title(f"Wealth Dist | Step {t}")
        axes[0].set_xlim(lo, hi)

        for m, line in lines.items():
            line.set_data(times, series[m])
        
        visible_data = [series[m] for m, l in lines.items() if l.get_visible() and series[m]]
        if visible_data:
            ymax = max(max(d) for d in visible_data)
            axes[1].set_ylim(0, max(ymax * 1.1, 1e-3))

        return list(lines.values())

    ani = animation.FuncAnimation(fig, update, frames=steps // sample_every, interval=30, blit=False, repeat=False)
    plt.show()
    return ani


def main():
    # --- 年化參數設定 (Annual Parameters) ---
    simulation_years = 30           # 模擬年數
    steps_per_year = 1000           # 一年拆成多少步 (動)
    
    annual_labor_income = 0.5       # 每年基礎勞動收入 (相對於初始財富 1.0)
    annual_labor_vol = 0.05         # 勞動收入年化波動率
    annual_tax_rate_labor = 0.4   # 20% 年化勞動所得稅
    annual_tax_rate_capital = 0.5  # 2% 年化資產稅 (如房稅/遺產稅平攤)
    annual_capital_return = 0.05    # 5% 年化資本增長率 (錢滾錢)
    
    high_skill_ratio = 0.2          # 20% 高技術人口
    high_skill_bonus = 2.0          # 高技術者收入為 2 倍
    
    # --- 交換行為參數 (維度無關) ---
    saving = 0.3                    # 基礎儲蓄率
    saving_slope = 0.4              # 儲蓄隨財富增加的傾向
    saving_min = 0.1
    saving_max = 0.95
    
    # --- 執行參數 ---
    n_agents = 1000
    metrics = ["gini", "top10_share"]
    sample_every_n_days = 30        # 大約每隔「一個月」取樣一次繪圖
    
    # --- 參數標準化 (Normalization to per-step) ---
    total_steps = simulation_years * steps_per_year
    
    # 勞動收入平攤到每步
    labor_income_step = annual_labor_income / steps_per_year
    # 波動率縮放 (統計原理：標準差隨步數平方根縮放)
    labor_vol_step = annual_labor_vol / np.sqrt(steps_per_year)
    # 資本回報率 (複利反推：(1+r_step)^steps = 1+r_annual)
    capital_return_step = (1.0 + annual_capital_return)**(1.0/steps_per_year) - 1.0
    # 稅率標準化
    tax_rate_labor_step = annual_tax_rate_labor # 所得稅率不隨頻率改變，因為它是按次扣除
    tax_rate_capital_step = annual_tax_rate_capital / steps_per_year # 財產稅平攤到每一步
    
    sample_every = int(steps_per_year * (sample_every_n_days / 365.0))
    if sample_every < 1: sample_every = 1

    print(f"--- 啟動模擬 ---")
    print(f"模擬總長: {simulation_years} 年 | 總步數: {total_steps} | 每步代表: {365/steps_per_year:.2f} 天")
    print(f"年化參數: 勞收={annual_labor_income}, 資本利得={annual_capital_return*100}%, 稅率={annual_tax_rate_labor*100}%/{annual_tax_rate_capital*100}%")

    params = {
        "n_agents": n_agents,
        "steps": total_steps,
        "saving": saving,
        "saving_slope": saving_slope,
        "saving_min": saving_min,
        "saving_max": saving_max,
        "sample_every": sample_every,
        "tax_rate_labor": tax_rate_labor_step,
        "tax_rate_capital": tax_rate_capital_step,
        "high_skill_ratio": high_skill_ratio,
        "labor_income": labor_income_step,
        "labor_vol": labor_vol_step,
        "capital_return": capital_return_step,
        "high_skill_bonus": high_skill_bonus,
        "seed": 42,
        "metrics": metrics
    }

    global _persistent_ani
    _persistent_ani = run_animation(**params)


if __name__ == "__main__":
    main()
