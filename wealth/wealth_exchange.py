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
    labor_growth_step: float = 0.0,
):
    rng = np.random.default_rng(seed)
    w = np.ones(n_agents, dtype=float)
    high_skill = rng.random(n_agents) < high_skill_ratio

    all_possible_metrics = ["gini", "mean", "median", "p90_p10", "top10_share", "labor_share"]
    series = {m: [] for m in all_possible_metrics}
    times = []

    current_labor_income = labor_income
    total_labor_added = 0.0
    total_capital_added = 0.0

    for t in range(steps):
        # 0. GDP 成長
        if labor_growth_step > 0:
            current_labor_income *= (1.0 + labor_growth_step)

        # 1. 隨機交換
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

        # 2. 勞動收入
        if current_labor_income > 0.0:
            wages = np.full(n_agents, current_labor_income, dtype=float)
            if labor_vol > 0.0:
                current_vol = labor_vol * (current_labor_income / labor_income)
                wages += rng.normal(0.0, current_vol, size=n_agents)
                wages = np.clip(wages, 0.0, None)
            if high_skill_ratio > 0.0:
                wages[high_skill] *= high_skill_bonus
            
            labor_sum = np.sum(wages)
            total_labor_added += labor_sum
            
            if tax_rate_labor > 0.0:
                tax = wages * tax_rate_labor
                wages -= tax
                w += wages + np.sum(tax) / n_agents
            else:
                w += wages

        # 3. 資本回報
        if capital_return > 0.0:
            old_w_sum = np.sum(w)
            w *= (1.0 + capital_return)
            total_capital_added += np.sum(w) - old_w_sum
            
            if tax_rate_capital > 0.0:
                tax = w * tax_rate_capital
                w = (w - tax) + np.sum(tax) / n_agents

        if not np.all(np.isfinite(w)):
            w = np.nan_to_num(w, nan=0.0, posinf=1e15, neginf=0.0)

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
            combined_added = (total_labor_added + total_capital_added) or 1.0
            series["labor_share"].append(float(total_labor_added / combined_added))

    return w, times, series


def run_animation(
    n_agents: int, steps: int, saving: float, seed: int,
    saving_slope: float, saving_min: float, saving_max: float,
    sample_every: int, metrics: list[str], tax_rate_labor: float,
    tax_rate_capital: float, high_skill_ratio: float,
    labor_income: float, labor_vol: float, capital_return: float,
    high_skill_bonus: float, labor_growth_step: float = 0.0
):
    rng = np.random.default_rng(seed)
    w = np.ones(n_agents, dtype=float)
    high_skill = rng.random(n_agents) < high_skill_ratio
    current_labor_income = labor_income
    total_labor_added = 0.0
    total_capital_added = 0.0

    all_metrics = ["gini", "mean", "median", "p90_p10", "top10_share", "labor_share"]
    series = {m: [] for m in all_metrics}
    times = []

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(right=0.85)

    lines = {}
    for m in all_metrics:
        line, = axes[1].plot([], [], label=m)
        line.set_visible(m in metrics)
        lines[m] = line
    
    axes[1].set_title("Economic Metrics Over Time")
    axes[1].set_xlabel("Steps")
    axes[1].set_xlim(0, steps)
    axes[1].legend(loc="upper left", fontsize='small')

    check_ax = fig.add_axes([0.88, 0.2, 0.1, 0.6])
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
        nonlocal w, current_labor_income, total_labor_added, total_capital_added
        for _ in range(sample_every):
            if labor_growth_step > 0:
                current_labor_income *= (1.0 + labor_growth_step)

            i, j = rng.integers(0, n_agents, size=2)
            if i != j:
                wi, wj = w[i], w[j]
                mean_w = np.mean(w)
                si = np.clip(saving + saving_slope * (wi/(mean_w or 1.0) - 1.0), saving_min, saving_max)
                sj = np.clip(saving + saving_slope * (wj/(mean_w or 1.0) - 1.0), saving_min, saving_max)
                traded = (wi - si*wi) + (wj - sj*wj)
                eps = rng.random()
                w[i], w[j] = si*wi + eps*traded, sj*wj + (1.0-eps)*traded
            
            if current_labor_income > 0:
                wages = np.full(n_agents, current_labor_income) + rng.normal(0, labor_vol * (current_labor_income/labor_income), n_agents)
                wages = np.clip(wages, 0, None)
                wages[high_skill] *= high_skill_bonus
                labor_sum = np.sum(wages)
                total_labor_added += labor_sum
                tax_l = wages * tax_rate_labor
                w += (wages - tax_l) + np.sum(tax_l)/n_agents
            
            if capital_return > 0:
                old_sum = np.sum(w)
                w *= (1.0 + capital_return)
                total_capital_added += np.sum(w) - old_sum
                tax_c = w * tax_rate_capital
                w = (w - tax_c) + np.sum(tax_c)/n_agents
            
            if not np.all(np.isfinite(w)):
                w = np.nan_to_num(w, posinf=1e12, nan=0)

        t = (frame + 1) * sample_every
        times.append(t)
        w_sorted = np.sort(w)
        sum_w = np.sum(w_sorted) or 1.0
        series["gini"].append(gini(w_sorted))
        series["mean"].append(float(np.mean(w_sorted)))
        series["median"].append(float(np.median(w_sorted)))
        p90, p10 = np.percentile(w_sorted, [90, 10])
        series["p90_p10"].append(float(p90 / max(p10, 1e-9)))
        series["top10_share"].append(float(np.sum(w_sorted[-int(0.1*n_agents):]) / sum_w))
        combined_added = (total_labor_added + total_capital_added) or 1.0
        series["labor_share"].append(float(total_labor_added / combined_added))

        axes[0].cla()
        lo, hi = np.percentile(w, [0, 99])
        if hi <= lo: hi = lo + 1.0
        axes[0].hist(np.clip(w, lo, hi), bins=40, density=True, color="#2a6f97", edgecolor="white")
        axes[0].set_title(f"Wealth Dist | Step {t}\nAvg Annual Wage: {current_labor_income * (steps/30/1000 * 1000):.2f}")
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
    # --- 社會契約設定 (Social Contract Parameters) ---
    simulation_years = 30           
    steps_per_year = 1000           
    
    annual_gdp_growth = 0.07        # 年化生產力成長
    annual_labor_income = 0.5       # 基礎年薪 (相對於初始資產 1.0)
    annual_labor_vol = 0.05         # 薪資穩定度 (波動率)
    annual_tax_rate_labor = 0.20    # 所得稅
    annual_tax_rate_capital = 0.05  # 財產稅
    annual_capital_return = 0.05    # 資本利得率
    
    high_skill_ratio = 0.2          
    high_skill_bonus = 2.0          
    
    saving_base = 0.2               # 基礎儲蓄率 (預設存多少比例)
    saving_slope = 0.2              # 階級儲蓄傾向
    saving_min = 0.1                # 儲蓄下限 (窮人保命錢)
    saving_max = 0.90               # 儲蓄上限 (富人投資極限)
    
    # --- 顯示與啟動提示 ---
    print(f"\n" + "="*40)
    print(f"      【財富分配模擬器 - 社會環境設定】")
    print(f"="*40)
    print(f" 模擬期間: {simulation_years} 年")
    print(f" 基礎年薪: {annual_labor_income} (起始資產的 {annual_labor_income*100:.0f}%)")
    print(f" 薪資環境: {'穩定' if annual_labor_vol < 0.1 else '劇烈波動'} (波動率: ±{annual_labor_vol})")
    print(f" 生產力成長: {annual_gdp_growth*100}% / 每年")
    print(f" 財富重分配: 勞動稅 {annual_tax_rate_labor*100}% | 資本稅 {annual_tax_rate_capital*100}%")
    print(f" 儲蓄行為: 基礎 {saving_base*100}% | 斜率 {saving_slope}")
    print(f" 儲蓄上下限: {saving_min*100}% ~ {saving_max*100}%")
    print(f"="*40 + "\n")

    # --- 標準化 ---
    total_steps = simulation_years * steps_per_year
    labor_income_step = annual_labor_income / steps_per_year
    labor_growth_step = (1.0 + annual_gdp_growth)**(1.0/steps_per_year) - 1.0
    labor_vol_step = annual_labor_vol / np.sqrt(steps_per_year)
    capital_return_step = (1.0 + annual_capital_return)**(1.0/steps_per_year) - 1.0
    
    sample_every = int(steps_per_year * (30 / 365.0)) # 每月取樣
    if sample_every < 1: sample_every = 1

    params = {
        "n_agents": 1000,
        "steps": total_steps,
        "saving": saving_base,
        "saving_slope": saving_slope,
        "saving_min": saving_min,
        "saving_max": saving_max,
        "sample_every": sample_every,
        "tax_rate_labor": annual_tax_rate_labor,
        "tax_rate_capital": annual_tax_rate_capital / steps_per_year,
        "high_skill_ratio": high_skill_ratio,
        "labor_income": labor_income_step,
        "labor_vol": labor_vol_step,
        "capital_return": capital_return_step,
        "high_skill_bonus": high_skill_bonus,
        "labor_growth_step": labor_growth_step,
        "seed": 42,
        "metrics": ["gini", "top10_share", "labor_share"]
    }

    global _persistent_ani
    _persistent_ani = run_animation(**params)


if __name__ == "__main__":
    main()
