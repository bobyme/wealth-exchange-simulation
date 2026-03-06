import numpy as np
import matplotlib.pyplot as plt
from wealth_exchange import simulate_exchange, gini

def save_model_plot(name, title, params):
    # 計算每步參數
    steps_per_year = 1000
    total_steps = 30 * steps_per_year
    
    # 標準化
    labor_income_step = params["annual_labor_income"] / steps_per_year
    labor_vol_step = 0.1 / np.sqrt(steps_per_year)
    capital_return_step = (1.0 + params["annual_capital_return"])**(1.0/steps_per_year) - 1.0
    tax_rate_capital_step = params["annual_tax_rate_capital"] / steps_per_year
    
    # 執行模擬 (使用修改後的核心邏輯)
    w, times, series = simulate_exchange(
        n_agents=1000,
        steps=total_steps,
        saving=0.3,
        saving_slope=params["saving_slope"],
        saving_min=0.1,
        saving_max=0.95,
        sample_every=steps_per_year, # 每年取樣一次
        tax_rate_labor=params["annual_tax_rate_labor"],
        tax_rate_capital=tax_rate_capital_step,
        high_skill_ratio=0.2,
        labor_income=labor_income_step,
        labor_vol=labor_vol_step,
        capital_return=capital_return_step,
        high_skill_bonus=params["high_skill_bonus"],
        seed=42
    )
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 分佈圖
    lo, hi = np.percentile(w, [0, 99])
    axes[0].hist(np.clip(w, lo, hi), bins=40, density=True, color="#2a6f97", edgecolor="white")
    axes[0].set_title(f"{title} - Wealth Dist (Year 30)")
    axes[0].set_xlabel("Wealth")
    
    # 指標趨勢
    years = np.array(times) / steps_per_year
    axes[1].plot(years, series["gini"], label="Gini", lw=2)
    axes[1].plot(years, series["top10_share"], label="Top 10% Share", lw=2)
    axes[1].set_title(f"{title} - Inequality Metrics")
    axes[1].set_xlabel("Years")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()
    return series["gini"][-1]

# 執行三種模式
models = {
    "nordic": {
        "title": "Nordic Model (Welfare State)",
        "annual_labor_income": 0.8,
        "annual_tax_rate_labor": 0.45,
        "annual_tax_rate_capital": 0.10,
        "annual_capital_return": 0.04,
        "high_skill_bonus": 1.5,
        "saving_slope": 0.2
    },
    "taiwan": {
        "title": "Taiwan Model (Balanced)",
        "annual_labor_income": 0.4,
        "annual_tax_rate_labor": 0.15,
        "annual_tax_rate_capital": 0.02,
        "annual_capital_return": 0.06,
        "high_skill_bonus": 2.5,
        "saving_slope": 0.5
    },
    "usa": {
        "title": "USA Model (Laissez-faire)",
        "annual_labor_income": 0.3,
        "annual_tax_rate_labor": 0.25,
        "annual_tax_rate_capital": 0.05,
        "annual_capital_return": 0.09, # 高風險溢酬
        "high_skill_bonus": 4.5,
        "saving_slope": 0.7
    }
}

results = {}
for name, params in models.items():
    print(f"Running {name}...")
    final_gini = save_model_plot(name, params["title"], params)
    results[name] = final_gini

print("Simulations complete.")
