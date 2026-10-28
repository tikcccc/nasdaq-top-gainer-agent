# purpose: Compute 5-day returns, trend, volatility, drawdown and benchmark comparison for AIOK

import pandas as pd, numpy as np, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

stock = load("ds_AIOK")
gspc = load("ds_GSPC")
ixic = load("ds_IXIC")

def cum(df):
    c = df["close"].astype(float)
    return float(c.iloc[-1] / c.iloc[0] - 1)

c = stock["close"].astype(float).reset_index(drop=True)
rets = [float(c.iloc[i] / c.iloc[i-1] - 1) for i in range(1, len(c))]
daily_returns = rets[-5:]
avg_daily_return = float(np.mean(daily_returns))
cumulative_return = float(c.iloc[-1] / c.iloc[0] - 1)

# trend from closes
x = np.arange(len(c))
norm = c / c.iloc[0]
slope = float(np.polyfit(x, norm, 1)[0])
if cumulative_return > 0.05 and slope > 0:
    trend = "up"
elif cumulative_return < -0.05 and slope < 0:
    trend = "down"
else:
    trend = "sideways"

vol_daily = float(np.std(daily_returns, ddof=1))
vol_ann = vol_daily * math.sqrt(252)

# max drawdown
peak = -np.inf; mdd = 0.0
for v in c:
    if v > peak: peak = v
    dd = v / peak - 1
    if dd < mdd: mdd = float(dd)

bench = {"GSPC": cum(gspc), "IXIC": cum(ixic)}
excess = cumulative_return - bench["GSPC"]
best_day = max(daily_returns); worst_day = min(daily_returns)

save_result({
    "daily_returns": daily_returns,
    "avg_daily_return": avg_daily_return,
    "cumulative_return": cumulative_return,
    "trend": trend,
    "trend_slope": slope,
    "volatility_daily": vol_daily,
    "volatility_annualized": vol_ann,
    "max_drawdown": mdd,
    "benchmark": bench,
    "excess_return": excess,
    "best_day": best_day,
    "worst_day": worst_day,
    "notes": "Window 2026-10-16 to 2026-10-23; sharp single-day surge on the final session."
})

# chart: all indexed to 100 at window start
fig, ax = plt.subplots(figsize=(8, 4.5))
for df, label in [(stock, "AIOK"), (gspc, "S&P 500"), (ixic, "NASDAQ Comp")]:
    v = df["close"].astype(float).reset_index(drop=True)
    idx = v / v.iloc[0] * 100
    ax.plot(range(len(idx)), idx, marker="o", label=label)
ax.set_title("AIOK vs Benchmarks - Last 5 Trading Days (indexed to 100)")
ax.set_xlabel("Day (2026-10-16 -> 2026-10-23)")
ax.set_ylabel("Index (start = 100)")
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
save_chart(fig)
print("done", cumulative_return, trend, bench)
