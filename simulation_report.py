"""
美國 vs 台灣居住隔離模擬報告
使用 Schelling Segregation Model（含 CBD 效應改良版）
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # 非互動模式，直接存檔
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import convolve
from datetime import date

from human_simulation import Params, init_grid, step

# 支援中文字型（macOS）
plt.rcParams["font.sans-serif"] = ["PingFang TC", "Heiti TC", "Arial Unicode MS", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIR = "report_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CMAP = ListedColormap(["#1e1e1e", "#20B2AA", "#FFD700"])


# ── 統計工具 ────────────────────────────────────────────────────────────────

def compute_segregation_index(grid: np.ndarray, p: Params) -> float:
    """
    隔離指數：所有有住戶格子的「同族鄰居比例」平均值。
    0.5 = 完全隨機混居；1.0 = 完全隔離。
    """
    if p.neighborhood == "von_neumann":
        kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    else:
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)

    same1 = convolve((grid == 1).astype(float), kernel, mode="constant", cval=0)
    same2 = convolve((grid == 2).astype(float), kernel, mode="constant", cval=0)
    occ_n = convolve((grid != 0).astype(float), kernel, mode="constant", cval=0)

    mask1 = (grid == 1) & (occ_n > 0)
    mask2 = (grid == 2) & (occ_n > 0)

    ratios = []
    if mask1.any():
        ratios.extend((same1[mask1] / occ_n[mask1]).tolist())
    if mask2.any():
        ratios.extend((same2[mask2] / occ_n[mask2]).tolist())

    return float(np.mean(ratios)) if ratios else 0.0


# ── 模擬執行 ────────────────────────────────────────────────────────────────

def run_scenario(name: str, p: Params, seed: int = 42) -> dict:
    print(f"  模擬中：{name} ...")
    rng = np.random.default_rng(seed)
    grid = init_grid(p, seed=seed)

    init_seg = compute_segregation_index(grid, p)
    history_unhappy = []

    for _ in range(p.max_steps):
        grid, moves, unhappy_n = step(grid, p, rng)
        history_unhappy.append(unhappy_n)
        if unhappy_n == 0 or moves == 0:
            break

    steps_taken = len(history_unhappy)
    final_seg = compute_segregation_index(grid, p)
    final_unhappy = history_unhappy[-1] if history_unhappy else 0
    total_occupied = int((grid != 0).sum())

    # 最終分布圖
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    if p.cbd_gravity > 0:
        c = p.size / 2.0
        ax.add_patch(plt.Circle((c, c), p.size * 0.2, color="red",
                                 fill=False, linestyle="--", linewidth=2, alpha=0.6))
        ax.plot(c, c, "r+", markersize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name} — 最終狀態（{steps_taken} 步）", fontsize=13)
    img_path = os.path.join(OUTPUT_DIR, f"{name}_final.png")
    fig.savefig(img_path, dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 收斂曲線圖
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    ax2.plot(range(1, steps_taken + 1), history_unhappy, color="#e05c5c", linewidth=1.5)
    ax2.fill_between(range(1, steps_taken + 1), history_unhappy, alpha=0.15, color="#e05c5c")
    ax2.set_xlabel("步數")
    ax2.set_ylabel("不滿意人數")
    ax2.set_title(f"{name} — 不滿意人數收斂曲線")
    ax2.grid(True, alpha=0.3)
    conv_path = os.path.join(OUTPUT_DIR, f"{name}_convergence.png")
    fig2.savefig(conv_path, dpi=130, bbox_inches="tight")
    plt.close(fig2)

    print(f"    完成：{steps_taken} 步 | 隔離指數 {init_seg:.3f} → {final_seg:.3f} "
          f"| 剩餘不滿意 {final_unhappy}/{total_occupied}")

    return {
        "name": name,
        "steps": steps_taken,
        "init_seg": init_seg,
        "final_seg": final_seg,
        "delta_seg": final_seg - init_seg,
        "final_unhappy": final_unhappy,
        "total_occupied": total_occupied,
        "img_path": img_path,
        "conv_path": conv_path,
        "params": p,
    }


# ── 報告生成 ────────────────────────────────────────────────────────────────

def write_report(results: list) -> str:
    today = date.today().isoformat()
    lines = []

    lines += [
        f"# 居住隔離模擬報告：美國 vs 台灣",
        f"",
        f"生成日期：{today}  ",
        f"模型：Schelling Segregation Model（改良版，含 CBD 市中心效應）",
        f"",
        f"---",
        f"",
        f"## 模型說明",
        f"",
        f"本模擬基於 Thomas Schelling（1971）提出的居住隔離模型：",
        f"每位居民若「同族鄰居比例」低於個人門檻，即視為不滿意並嘗試搬家。",
        f"即使門檻設定相當溫和，群體動態仍會放大成明顯的社會隔離現象。",
        f"",
        f"### 改良參數",
        f"",
        f"| 參數 | 意義 | 現實對應 |",
        f"|------|------|----------|",
        f"| `friction_cost` | 搬家阻力（0=無阻力，1=完全無法搬） | 房貸綁定、學區成本、交易稅 |",
        f"| `cbd_gravity` | 市中心吸引力（0=無，1=全往中心跑） | 捷運地段競爭、都市包容度 |",
        f"",
        f"### 隔離指數說明",
        f"",
        f"- **0.50**：完全隨機混居（理論下限）",
        f"- **0.70**：輕度隔離",
        f"- **0.85+**：強烈隔離板塊",
        f"",
        f"---",
        f"",
    ]

    for r in results:
        p = r["params"]
        lines += [
            f"## {r['name']}",
            f"",
            f"### 參數設定與設計理由",
            f"",
            f"| 參數 | 值 | 設計理由 |",
            f"|------|----|----------|",
            f"| `size` | {p.size} | {p.size}×{p.size} 網格 |",
            f"| `empty_ratio` | {p.empty_ratio} | 空屋率 {p.empty_ratio*100:.0f}% |",
            f"| `group1_ratio` | {p.group1_ratio} | 多數群體 {p.group1_ratio*100:.0f}% / 少數 {(1-p.group1_ratio)*100:.0f}% |",
            f"| `threshold_g1` | {p.threshold_g1} | 多數群體滿意門檻 |",
            f"| `threshold_g2` | {p.threshold_g2} | 少數群體滿意門檻 |",
            f"| `friction_cost` | {p.friction_cost} | 搬家阻力 |",
            f"| `cbd_gravity` | {p.cbd_gravity} | 市中心容忍度加成 |",
            f"| `cbd_gravity_g1` | {p.cbd_gravity_g1:.2f} | Group1（窮人）搬家市中心偏好 |",
            f"| `cbd_gravity_g2` | {p.cbd_gravity_g2:.2f} | Group2（富人）搬家市中心偏好 |",
            f"| `neighborhood` | {p.neighborhood} | 8宮格鄰居 |",
            f"| `max_steps` | {p.max_steps} | 最大步數 |",
            f"",
            f"### 模擬結果",
            f"",
            f"| 指標 | 數值 |",
            f"|------|------|",
            f"| 收斂步數 | {r['steps']} 步 |",
            f"| 初始隔離指數 | {r['init_seg']:.3f} |",
            f"| 最終隔離指數 | {r['final_seg']:.3f} |",
            f"| 隔離指數上升 | +{r['delta_seg']:.3f} |",
            f"| 最終不滿意人數 | {r['final_unhappy']} / {r['total_occupied']} 人 "
            f"（{r['final_unhappy']/r['total_occupied']*100:.1f}%）|",
            f"",
            f"### 最終居住分布",
            f"",
            f"![{r['name']} 最終分布]({r['name']}_final.png)",
            f"",
            f"> 藍綠色 = 群體 1（多數）；金色 = 群體 2（少數）；黑色 = 空屋",
            f"> 紅色虛線圓圈 = CBD 市中心容忍加成範圍",
            f"",
            f"### 不滿意人數收斂曲線",
            f"",
            f"![{r['name']} 收斂曲線]({r['name']}_convergence.png)",
            f"",
            f"---",
            f"",
        ]

    # 比較表
    usa = next(r for r in results if "美國" in r["name"])
    twn = next(r for r in results if "台灣" in r["name"])

    lines += [
        f"## 兩國比較",
        f"",
        f"| 指標 | 美國 | 台灣 |",
        f"|------|------|------|",
        f"| 收斂步數 | {usa['steps']} | {twn['steps']} |",
        f"| 初始隔離指數 | {usa['init_seg']:.3f} | {twn['init_seg']:.3f} |",
        f"| 最終隔離指數 | {usa['final_seg']:.3f} | {twn['final_seg']:.3f} |",
        f"| 隔離指數上升幅度 | +{usa['delta_seg']:.3f} | +{twn['delta_seg']:.3f} |",
        f"| 最終不滿意比例 | {usa['final_unhappy']/usa['total_occupied']*100:.1f}% | {twn['final_unhappy']/twn['total_occupied']*100:.1f}% |",
        f"| 搬家阻力 | {usa['params'].friction_cost} （低） | {twn['params'].friction_cost} （高） |",
        f"| 市中心吸引力 | {usa['params'].cbd_gravity} （弱） | {twn['params'].cbd_gravity} （強） |",
        f"",
        f"---",
        f"",
        f"## 結論分析",
        f"",
        f"### 美國",
        f"",
        f"- **低搬家阻力**（{usa['params'].friction_cost}）：居民對不滿意環境快速反應，隔離板塊形成速度快",
        f"- **弱 CBD 效應**（{usa['params'].cbd_gravity}）：郊區化傾向使隔離從城市邊緣蔓延，缺乏混居緩衝區",
        f"- **多數/少數比 {usa['params'].group1_ratio*100:.0f}%/{(1-usa['params'].group1_ratio)*100:.0f}%**：少數群體較難在各處形成足夠規模的聚落，被擠壓到少數角落",
        f"- 最終隔離指數 **{usa['final_seg']:.3f}**，呈現清晰的大型單色板塊",
        f"",
        f"### 台灣",
        f"",
        f"- **高搬家阻力**（{twn['params'].friction_cost}）：房貸綁定、學區成本使居民即使不滿意也傾向忍耐，隔離演化較慢",
        f"- **強 CBD 效應（非對稱）**：富人 cbd_gravity={twn['params'].cbd_gravity_g2}，窮人 cbd_gravity={twn['params'].cbd_gravity_g1:.2f}",
        f"  → 富人優先搶佔市中心，窮人被推向外圍郊區，形成「中心富人、外圍窮人」的同心圓結構",
        f"- **高搬家阻力**（{twn['params'].friction_cost}）：即使不滿意，仍有 {twn['params'].friction_cost*100:.0f}% 機率忍耐不動，隔離演化較慢",
        f"- 最終隔離指數 **{twn['final_seg']:.3f}**，隔離現象較美國緩和，但具有明顯的經濟階層空間分化",
        f"",
        f"### Schelling 核心洞察",
        f"",
        f"兩個場景都印證了 Schelling 的原始發現：",
        f"**個人溫和的偏好（門檻 50~65%），在群體動態下會放大成遠超預期的社會隔離結構。**",
        f"政策意涵上，單純降低門檻（提倡包容）的效果有限；",
        f"提高搬家阻力（租金補貼、稅制改革）或強化 CBD 混居誘因，",
        f"才能從機制層面減緩隔離的自發性形成。",
    ]

    report_path = os.path.join(OUTPUT_DIR, "segregation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return report_path


# ── 主程式 ──────────────────────────────────────────────────────────────────

SCENARIOS = [
    ("美國（種族隔離）", Params(
        size=60,
        empty_ratio=0.15,    # 美國空屋率約 11~15%
        group1_ratio=0.60,   # 多數族群 60% / 少數族群 40%
        threshold_g1=0.65,   # 歷史上較高的種族偏好門檻
        threshold_g2=0.65,   # 少數族群同樣傾向防禦性聚居
        friction_cost=0.15,  # 美國流動率高，搬家阻力低
        cbd_gravity=0.2,     # 郊區化，市中心吸引力弱（兩群相同）
        max_steps=300,
        neighborhood="moore",
    )),
    ("台灣（階級隔離）", Params(
        size=60,
        empty_ratio=0.15,    # 台灣整體空屋率約 10~15%
        group1_ratio=0.65,   # Group1=窮人 65% / Group2=富人 35%
        threshold_g1=0.55,   # 窮人對混居有一定容忍度
        threshold_g2=0.55,   # 富人亦然，但搬家行為不同
        friction_cost=0.50,  # 高搬家阻力：房貸綁定、高房價、學區限制
        cbd_gravity=0.85,    # 市中心容忍度加成（兩群都更包容）
        cbd_gravity_g1=0.2,  # 窮人：負擔不起市中心，隨機找郊區空地
        cbd_gravity_g2=0.95, # 富人：強力搶佔市中心精華地段
        max_steps=500,
        neighborhood="moore",
    )),
]

if __name__ == "__main__":
    print("=== 居住隔離模擬：美國 vs 台灣 ===\n")

    results = []
    for name, p in SCENARIOS:
        r = run_scenario(name, p, seed=42)
        results.append(r)

    print("\n輸出報告...")
    report_path = write_report(results)
    print(f"\n完成！報告輸出至：{report_path}")
    print(f"圖片存放於：{OUTPUT_DIR}/")
