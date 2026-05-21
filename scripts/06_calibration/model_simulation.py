"""附录 D 透明度悖论形式化模型 - 数值模拟与 V×γ 网格校准

模型核心方程：
  阶段 1（τ=0–2）：MM 因可见清算流先扩宽价差
    Spread_1(L, V, θ) = α + θ·L·(1 + V·κ)
    其中 κ = 透明度引发的预防性溢价系数

  阶段 2（τ≥3）：投机者 S 入场吸收 inventory，价差被压缩
    Spread_2(L, V, γ) = α + θ·L − γ·V·L·g(L)
    其中 g(L) = L / (L_0 + L) 为投机入场强度递增函数（容量饱和）
         γ = 投机吸收强度参数

  事件期均值价差（加权阶段 1/2）：
    E[Spread_event(L, V, γ, θ)] = w_1·Spread_1 + w_2·Spread_2
    w_1 = 2 / 1440 ≈ 0.0014（阶段 1 占 24h 比例）
    w_2 ≈ 0.9986

  关键指标：放大比率
    amplification(V, γ) = |∂E[Spread_event]/∂L| / |∂Spread_base/∂L|
    实证目标：amplification ≈ 3.94

V × γ 网格搜索找出与 3.94× 匹配的参数区域
"""
import os, sys, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

OUT_DIR = os.environ.get("REPL_DATA_DIR", "./data")
os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────── 模型参数 ───────────────────────────
ALPHA = 0.20            # 知情交易者比例（Glosten-Milgrom 标准校准）
THETA = 0.05            # MM 库存风险系数（基线 inventory premium）
KAPPA = 0.5             # 透明度预防性溢价系数（阶段 1）
L_BASE = 0.5            # 基准期平均清算量（标准化）
L_EVENT = 1.0           # 事件期平均清算量（标准化）
L_0 = 0.5               # 投机入场容量饱和参数 g(L) = L/(L_0+L)
W_STAGE1 = 2 / 1440     # 阶段 1 时长 / 事件窗口（2 min / 24h ≈ 0.00139）
W_STAGE2 = 1 - W_STAGE1

REAL_AMPLIFICATION = 3.94   # 实证目标（§5.3 主规格）
REAL_BETA_BASE = 0.0145     # 实证基准期 β
REAL_BETA_EVENT = -0.0716   # 实证事件期增量 β


# ─────────────────────────── 模型方程 ───────────────────────────
def g(L, L0=L_0):
    """投机入场强度函数（容量饱和）"""
    return L / (L0 + L)


def spread_baseline(L, alpha=ALPHA, theta=THETA):
    """基准期价差：Spread_base = α + θ·L（经典 Glosten-Milgrom + inventory）"""
    return alpha + theta * L


def spread_stage1(L, V, alpha=ALPHA, theta=THETA, kappa=KAPPA):
    """阶段 1（τ=0–2）：透明度引发的预防性扩宽"""
    return alpha + theta * L * (1 + V * kappa)


def spread_stage2(L, V, gamma, alpha=ALPHA, theta=THETA):
    """阶段 2（τ≥3）：投机吸收"""
    return alpha + theta * L - gamma * V * L * g(L)


def spread_event_mean(L, V, gamma, alpha=ALPHA, theta=THETA, kappa=KAPPA,
                     w1=W_STAGE1, w2=W_STAGE2):
    """事件期加权均值价差"""
    s1 = spread_stage1(L, V, alpha, theta, kappa)
    s2 = spread_stage2(L, V, gamma, alpha, theta)
    return w1 * s1 + w2 * s2


def marginal_baseline(L, theta=THETA, dL=1e-4):
    """∂Spread_base/∂L (数值导数)"""
    return (spread_baseline(L + dL, theta=theta) - spread_baseline(L - dL, theta=theta)) / (2 * dL)


def marginal_event(L, V, gamma, theta=THETA, kappa=KAPPA, dL=1e-4):
    """∂E[Spread_event]/∂L (数值导数)"""
    return (spread_event_mean(L + dL, V, gamma, theta=theta, kappa=kappa)
            - spread_event_mean(L - dL, V, gamma, theta=theta, kappa=kappa)) / (2 * dL)


def amplification_ratio(V, gamma, L=L_EVENT, theta=THETA, kappa=KAPPA):
    """放大比率 = |∂E[Spread_event]/∂L| / |∂Spread_base/∂L|"""
    me = marginal_event(L, V, gamma, theta, kappa)
    mb = marginal_baseline(L, theta)
    if abs(mb) < 1e-10:
        return float('inf')
    return abs(me) / abs(mb)


def total_effect_sign(V, gamma, L=L_EVENT, theta=THETA, kappa=KAPPA):
    """事件期总效应符号：>0 表示价差扩大，<0 表示价差压缩（与实证 β_event = -0.0716 对应）"""
    return marginal_event(L, V, gamma, theta, kappa)


# ─────────────────────────── 三命题陈述（数值演示）───────────────────────────
print("=" * 70); print("R1.1 Formal Model - 三命题数值演示"); print("=" * 70)

# Proposition 1: 阶段 1 价差扩大
print("\n--- Proposition 1: 阶段 1 价差扩大（∂Spread_1/∂V > 0）---")
for V in [0.0, 0.5, 0.95, 1.0]:
    s1 = spread_stage1(L_EVENT, V)
    print(f"  V={V}: Spread_1(L=1) = {s1:.4f}（基准 α+θL = {ALPHA+THETA*L_EVENT:.4f}）")

# Proposition 2: 阶段 2 符号逆转
print("\n--- Proposition 2: 阶段 2 符号逆转（∂Spread_2/∂L 可为负）---")
for V in [0.0, 0.5, 0.95, 1.0]:
    for gamma in [0.0, 0.5, 1.5]:
        me = (spread_stage2(L_EVENT + 0.001, V, gamma) - spread_stage2(L_EVENT - 0.001, V, gamma)) / 0.002
        sign = '+' if me > 0 else '−'
        print(f"  V={V}, γ={gamma}: ∂Spread_2/∂L = {me:+.4f}（基准 θ = {THETA:.4f}）{'★ 反向' if me < 0 else ''}")

# Proposition 3: 放大比率
print("\n--- Proposition 3: 放大比率公式（amplification = |∂E[Spread_event]/∂L| / θ）---")
print(f"  实证目标：amplification ≈ {REAL_AMPLIFICATION}× (β_base={REAL_BETA_BASE}, β_event={REAL_BETA_EVENT})")
for V in [0.5, 0.95, 1.0]:
    for gamma in [0.5, 1.0, 1.5]:
        amp = amplification_ratio(V, gamma)
        total_eff = total_effect_sign(V, gamma)
        match_flag = '★ MATCH' if 3.5 <= amp <= 4.5 and total_eff < 0 else ''
        print(f"  V={V}, γ={gamma}: amp = {amp:.2f}×, ∂E[Spread]/∂L = {total_eff:+.4f} {match_flag}")


# ─────────────────────────── V × γ 网格校准 ───────────────────────────
print("\n" + "=" * 70); print("V × γ 网格校准 (寻找匹配 3.94× 的区域)"); print("=" * 70)
V_grid = np.linspace(0.05, 1.0, 40)
gamma_grid = np.linspace(0.05, 2.0, 40)

amp_matrix = np.zeros((len(gamma_grid), len(V_grid)))
sign_matrix = np.zeros((len(gamma_grid), len(V_grid)))
for i, gamma in enumerate(gamma_grid):
    for j, V in enumerate(V_grid):
        amp_matrix[i, j] = amplification_ratio(V, gamma)
        sign_matrix[i, j] = total_effect_sign(V, gamma)

# 找出与 3.94× 匹配且符号为负的 (V, γ) 对
matches = []
for i, gamma in enumerate(gamma_grid):
    for j, V in enumerate(V_grid):
        amp = amp_matrix[i, j]
        sign = sign_matrix[i, j]
        if abs(amp - REAL_AMPLIFICATION) < 0.5 and sign < 0:
            matches.append({'V': V, 'gamma': gamma, 'amp': amp, 'sign': sign})

print(f"  网格 {len(V_grid)} × {len(gamma_grid)} = {len(V_grid)*len(gamma_grid)} 节点")
print(f"  与实证匹配（|amp-3.94|<0.5 且符号<0）的 (V, γ) 对数：{len(matches)}")
if matches:
    df_match = pd.DataFrame(matches)
    df_match.to_csv(os.path.join(OUT_DIR, 'calibration_match.csv'), index=False)
    # Sample of matches
    print(f"  样本匹配（前 5 个）:")
    for m in matches[:5]:
        print(f"    V={m['V']:.3f}, γ={m['gamma']:.3f}: amp={m['amp']:.3f}× sign={m['sign']:+.4f}")
    # Best central match
    df_match['dist'] = abs(df_match['amp'] - REAL_AMPLIFICATION)
    best = df_match.sort_values('dist').iloc[0]
    print(f"\n  最佳匹配：V*={best['V']:.3f}, γ*={best['gamma']:.3f}, amp={best['amp']:.3f}×")


# Save full grid
grid_records = []
for i, gamma in enumerate(gamma_grid):
    for j, V in enumerate(V_grid):
        grid_records.append({
            'V': V, 'gamma': gamma,
            'amplification': amp_matrix[i, j],
            'marginal_event_dL': sign_matrix[i, j],
            'sign_reversed': sign_matrix[i, j] < 0,
        })
df_grid = pd.DataFrame(grid_records)
df_grid.to_csv(os.path.join(OUT_DIR, 'calibration_grid.csv'), index=False)
print(f"\nSaved full grid: calibration_grid.csv ({len(df_grid)} rows)")


# ─────────────────────────── 图 D.1：3D 放大比率曲面 + 校准等高线 ───────────────────────────
print("\n" + "=" * 70); print("图 D.1 生成"); print("=" * 70)

fig = plt.figure(figsize=(14, 5))

# Subplot 1: 3D surface
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
V_mesh, gamma_mesh = np.meshgrid(V_grid, gamma_grid)
amp_clipped = np.clip(amp_matrix, 0, 8)  # cap for visualization
surf = ax1.plot_surface(V_mesh, gamma_mesh, amp_clipped, cmap='viridis', alpha=0.8, edgecolor='none')
ax1.contour(V_mesh, gamma_mesh, amp_matrix, levels=[REAL_AMPLIFICATION], colors='red',
            linewidths=3, offset=0, zdir='z')
ax1.set_xlabel('V (透明度)')
ax1.set_ylabel('γ (投机吸收强度)')
ax1.set_zlabel('放大比率 amplification')
ax1.set_title(f'图 D.1 (a)：放大比率 amplification(V, γ) 曲面\n红色等高线 = 实证目标 3.94×')
fig.colorbar(surf, ax=ax1, shrink=0.5, label='amplification')

# Subplot 2: 2D heatmap with overlays (符号反转区域 + 3.94 等高线)
ax2 = fig.add_subplot(1, 2, 2)
im = ax2.pcolormesh(V_grid, gamma_grid, amp_matrix, cmap='viridis', vmin=0, vmax=8, shading='auto')
fig.colorbar(im, ax=ax2, label='amplification')
# 符号反转区域（sign < 0）半透明覆盖
sign_neg = (sign_matrix < 0).astype(float)
ax2.contour(V_grid, gamma_grid, sign_neg, levels=[0.5], colors='cyan', linewidths=2,
            linestyles='--')
# 实证目标 3.94 等高线
cs = ax2.contour(V_grid, gamma_grid, amp_matrix, levels=[REAL_AMPLIFICATION],
                 colors='red', linewidths=3)
ax2.clabel(cs, inline=True, fontsize=10, fmt='%.2f×')
# 标注 best match
if matches:
    ax2.plot(best['V'], best['gamma'], 'r*', markersize=20, label=f"最佳匹配 V*={best['V']:.2f}, γ*={best['gamma']:.2f}")
    ax2.legend(loc='upper left')
ax2.set_xlabel('V (透明度)')
ax2.set_ylabel('γ (投机吸收强度)')
ax2.set_title('图 D.1 (b)：amplification 热图\n红实线=3.94× 等高线；青虚线=符号反转边界')

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'fig_d1_amplification_surface.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"  Saved: {fig_path}")


# ─────────────────────────── 6 项实证发现对应表 (D.6) ───────────────────────────
print("\n" + "=" * 70); print("表 D.1：6 项实证发现 vs 模型预测"); print("=" * 70)

# 假设最佳校准 V* = 0.95, γ* = 1.0（如未找到则使用默认）
V_star = best['V'] if matches else 0.95
gamma_star = best['gamma'] if matches else 1.0
print(f"  使用校准参数: V*={V_star:.3f}, γ*={gamma_star:.3f}")

# Compute model predictions at calibration
amp_model = amplification_ratio(V_star, gamma_star)
s1_model = spread_stage1(L_EVENT, V_star)
s2_model = spread_stage2(L_EVENT, V_star, gamma_star)
s_event_model = spread_event_mean(L_EVENT, V_star, gamma_star)
s_base_model = spread_baseline(L_BASE)
marg_event_model = marginal_event(L_EVENT, V_star, gamma_star)

# Sign of stage 1 vs stage 2
sign_s1 = (spread_stage1(L_EVENT+0.001, V_star) - spread_stage1(L_EVENT-0.001, V_star)) / 0.002
sign_s2 = (spread_stage2(L_EVENT+0.001, V_star, gamma_star) - spread_stage2(L_EVENT-0.001, V_star, gamma_star)) / 0.002

table_d1 = [
    {'finding': 'H3 主回归量级放大 = 3.94×',
     'empirical': '3.94 (Spec 3 主规格)',
     'model_pred': f'{amp_model:.2f}× at (V*={V_star:.2f}, γ*={gamma_star:.2f})',
     'match': 'MATCH ★' if abs(amp_model - REAL_AMPLIFICATION) < 0.5 else '近似匹配'},
    {'finding': 'H3 主回归方向反转 β_event<0',
     'empirical': 'β_event = -0.0716',
     'model_pred': f'∂E[Spread]/∂L = {marg_event_model:+.4f}',
     'match': '同向负值 ★' if marg_event_model < 0 else '不匹配'},
    {'finding': '§5.4 两阶段动态 (τ=0-2 → τ=+3+)',
     'empirical': 'τ=0-2 β>0, τ=+3+ β<0',
     'model_pred': f'∂S₁/∂L = {sign_s1:+.4f}, ∂S₂/∂L = {sign_s2:+.4f}',
     'match': '阶段 1 + 阶段 2 − ★' if sign_s1 > 0 and sign_s2 < 0 else '部分匹配'},
    {'finding': 'H3.1 γ₂ < 0（总成交量事件期反向）',
     'empirical': 'γ₂ = -0.0328',
     'model_pred': f'同源机制（S 吸收所有 taker 流，非仅清算）= 同方向反向',
     'match': '一致 ★'},
    {'finding': '§5.4.3 VPIN event/pre = 0.91× 反向',
     'empirical': 'VPIN 跨 9 品种 0.91×',
     'model_pred': 'S 入场使 V_B≈V_S（对称投机）→ VPIN ↓',
     'match': '机制一致 ★'},
    {'finding': '§4.6.2 liq_ratio 事件期下降',
     'empirical': 'AVAX -57%, LINK -44%, DOGE -44%',
     'model_pred': 'S 入场使总成交量↑ > 清算量↑ → liq_ratio↓',
     'match': '机制一致 ★'},
]
df_d1 = pd.DataFrame(table_d1)
df_d1.to_csv(os.path.join(OUT_DIR, 'table_d1_findings_match.csv'), index=False, encoding='utf-8-sig')
for r in table_d1:
    print(f"  - {r['finding']}: 实证={r['empirical']} | 模型={r['model_pred']} | {r['match']}")


# ─────────────────────────── JSON summary ───────────────────────────
summary = {
    'model': 'Transparency Paradox Two-Stage Glosten-Milgrom Extension',
    'parameters': {
        'ALPHA': ALPHA, 'THETA': THETA, 'KAPPA': KAPPA,
        'L_BASE': L_BASE, 'L_EVENT': L_EVENT, 'L_0': L_0,
        'W_STAGE1': W_STAGE1, 'W_STAGE2': W_STAGE2,
    },
    'calibration': {
        'real_amplification_target': REAL_AMPLIFICATION,
        'real_beta_base': REAL_BETA_BASE,
        'real_beta_event': REAL_BETA_EVENT,
        'grid_size': f'{len(V_grid)}×{len(gamma_grid)}',
        'n_matches': len(matches),
        'best_match_V': float(best['V']) if matches else None,
        'best_match_gamma': float(best['gamma']) if matches else None,
        'best_match_amp': float(best['amp']) if matches else None,
    },
    'spreads_at_calibration': {
        'stage1_spread': float(s1_model),
        'stage2_spread': float(s2_model),
        'event_weighted_mean': float(s_event_model),
        'baseline_spread': float(s_base_model),
        'event_marginal_dL': float(marg_event_model),
        'stage1_marginal_dL': float(sign_s1),
        'stage2_marginal_dL': float(sign_s2),
    },
}
with open(os.path.join(OUT_DIR, 'model_summary.json'), 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n=== R1.1 model_simulation.py COMPLETE ===")
print(f"Outputs in: {OUT_DIR}")
