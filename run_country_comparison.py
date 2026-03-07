import numpy as np
import matplotlib.pyplot as plt
from wealth_exchange import simulate_exchange

def run_country_model(name, title, params):
    steps_per_year = 1000
    total_steps = 30 * steps_per_year
    
    # 參數標準化 (同步 wealth_exchange.py 邏輯)
    labor_income_step = params["annual_labor_income"] / steps_per_year
    labor_growth_step = (1.0 + params["annual_gdp_growth"])**(1.0/steps_per_year) - 1.0
    labor_vol_step = params["annual_labor_vol"] / np.sqrt(steps_per_year)
    capital_return_step = (1.0 + params["annual_capital_return"])**(1.0/steps_per_year) - 1.0
    tax_rate_capital_step = params["annual_tax_rate_capital"] / steps_per_year
    
    w, times, series = simulate_exchange(
        n_agents=1000,
        steps=total_steps,
        saving=params["saving_base"],
        saving_slope=params["saving_slope"],
        saving_min=params["saving_min"],
        saving_max=params["saving_max"],
        sample_every=steps_per_year,
        tax_rate_labor=params["annual_tax_rate_labor"],
        tax_rate_capital=tax_rate_capital_step,
        high_skill_ratio=0.2,
        labor_income=labor_income_step,
        labor_vol=labor_vol_step,
        capital_return=capital_return_step,
        high_skill_bonus=params["high_skill_bonus"],
        labor_growth_step=labor_growth_step,
        seed=42
    )
    
    # 繪圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 分佈圖
    lo, hi = np.percentile(w, [0, 99])
    axes[0].hist(np.clip(w, lo, hi), bins=40, density=True, color="#2a6f97", edgecolor="white")
    axes[0].set_title(f"{title}\nWealth Dist (Year 30)")
    
    # 趨勢圖
    years = np.array(times) / steps_per_year
    axes[1].plot(years, series["gini"], label="Gini", lw=2)
    axes[1].plot(years, series["top10_share"], label="Top 10%", lw=2)
    axes[1].plot(years, series["labor_share"], label="Labor Share", lw=2, linestyle='--')
    axes[1].set_title(f"{title}\nInequality & Labor Share")
    axes[1].set_ylim(0, 1.0)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.close()
    return series["gini"][-1], series["labor_share"][-1]

# 四國參數設定
country_params = {
    "usa": {
        "title": "USA (Capitalist Growth)",
        "annual_gdp_growth": 0.025,
        "annual_labor_income": 0.4,
        "annual_labor_vol": 0.1,
        "annual_tax_rate_labor": 0.20,
        "annual_tax_rate_capital": 0.15,
        "annual_capital_return": 0.08, # 高股市回報
        "high_skill_bonus": 4.0,       # 矽谷/華爾街效應
        "saving_base": 0.1,            # 消費文化
        "saving_slope": 0.6,           # 貧富儲蓄極端化
        "saving_min": 0.02,            # 窮人幾乎不存錢
        "saving_max": 0.98             # 富人極致避險
    },
    "taiwan": {
        "title": "Taiwan (High Savings & Low Tax)",
        "annual_gdp_growth": 0.03,
        "annual_labor_income": 0.45,
        "annual_labor_vol": 0.05,
        "annual_tax_rate_labor": 0.12,
        "annual_tax_rate_capital": 0.02, # 資本利得稅極輕
        "annual_capital_return": 0.06, 
        "high_skill_bonus": 2.5,
        "saving_base": 0.35,           # 華人儲蓄傳統
        "saving_slope": 0.3,           # 全民皆存錢
        "saving_min": 0.15,
        "saving_max": 0.90
    },
    "japan": {
        "title": "Japan (Stable & Compressed)",
        "annual_gdp_growth": 0.005,    # 低增長
        "annual_labor_income": 0.5,
        "annual_labor_vol": 0.03,      # 極度穩定
        "annual_tax_rate_labor": 0.35, # 高所得稅
        "annual_tax_rate_capital": 0.25, # 高遺產稅/資產稅
        "annual_capital_return": 0.02, 
        "high_skill_bonus": 1.5,       # 薪資壓縮
        "saving_base": 0.4,            # 高防禦性儲蓄
        "saving_slope": 0.1,           # 儲蓄行為平均
        "saving_min": 0.3,
        "saving_max": 0.85
    },
    "china": {
        "title": "China (High Growth & High Vol)",
        "annual_gdp_growth": 0.055,    # 超高成長
        "annual_labor_income": 0.6,
        "annual_labor_vol": 0.2,       # 內捲/競爭劇烈
        "annual_tax_rate_labor": 0.15,
        "annual_tax_rate_capital": 0.05,
        "annual_capital_return": 0.10, # 高風險資產回報
        "high_skill_bonus": 3.5, 
        "saving_base": 0.4,            # 高儲蓄習慣 (無安全感)
        "saving_slope": 0.4,
        "saving_min": 0.2,
        "saving_max": 0.95
    }
}

print("Starting Multi-Country Simulation (30 Years)...")
for code, p in country_params.items():
    print(f"Simulating {code.upper()}...")
    run_country_model(code, p["title"], p)
print("Finished. Reports generated.")
