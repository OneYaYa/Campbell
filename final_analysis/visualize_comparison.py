"""
可视化 CrudeBERT vs FinBERT(oil_news) 累计情感得分对比
风格参考: https://github.com/Captain-1337/Master-Thesis

方法:
  1. 每条新闻按情感赋值: positive=+1, negative=-1, neutral=0
  2. 按日汇总得到 Daily Sentiment Sum
  3. 计算累计和 (Cumulative Sentiment Score)
  4. 7天滚动平均平滑
  5. MinMax 缩放到 [0, 1]
  6. 在同一张图上绘制 CrudeBERT (绿) 和 FinBERT (红)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.preprocessing import MinMaxScaler
import os

# ============ 配置 ============
CSV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csv_results")
YEARS = range(2017, 2025)
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualization_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

ROLLING_WINDOW = 7  # 7天滚动平均窗口

# 字体 & 样式
plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['axes.unicode_minus'] = False
FONT_SIZE_LABEL = 20
FONT_SIZE_TICK = 16

# ============ 加载数据 ============
def load_all(prefix):
    """加载某一类的所有年份CSV，合并为一个DataFrame"""
    frames = []
    for year in YEARS:
        path = os.path.join(CSV_DIR, f"{prefix}_{year}_result.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["date"])
            frames.append(df)
            print(f"  Loaded {os.path.basename(path)}  ({len(df)} rows)")
    combined = pd.concat(frames, ignore_index=True)
    combined["sentiment"] = combined["sentiment"].str.lower()
    combined = combined.sort_values("date").reset_index(drop=True)
    return combined


def sentiment_to_score(sentiment):
    """将情感标签转换为数值: positive=+1, negative=-1, neutral=0"""
    mapping = {"positive": 1, "negative": -1, "neutral": 0}
    return mapping.get(sentiment, 0)


def compute_cumulative_sentiment(df):
    """
    计算累计情感得分序列:
      1. 赋值 score = sentiment_to_score(sentiment)
      2. 按日汇总 daily_sum
      3. 累计求和 cumulative
      4. 7天滚动平均
      5. MinMax 缩放到 [0, 1]
    返回: 以日期为索引的 Series
    """
    df = df.copy()
    df["score"] = df["sentiment"].apply(sentiment_to_score)
    df["date_only"] = df["date"].dt.date

    # 按日汇总
    daily_sum = df.groupby("date_only")["score"].sum()
    daily_sum.index = pd.to_datetime(daily_sum.index)
    daily_sum = daily_sum.sort_index()

    # 累计和
    cumulative = daily_sum.cumsum()

    # 7天滚动平均
    rolling_mean = cumulative.rolling(window=ROLLING_WINDOW, min_periods=1).mean()

    # MinMax 缩放到 [0, 1]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(rolling_mean.values.reshape(-1, 1)).flatten()

    return pd.Series(scaled, index=rolling_mean.index, name="scaled_cumulative")


def compute_raw_cumulative(df):
    """返回未缩放的累计情感得分（带7天滚动平均）"""
    df = df.copy()
    df["score"] = df["sentiment"].apply(sentiment_to_score)
    df["date_only"] = df["date"].dt.date
    daily_sum = df.groupby("date_only")["score"].sum()
    daily_sum.index = pd.to_datetime(daily_sum.index)
    daily_sum = daily_sum.sort_index()
    cumulative = daily_sum.cumsum()
    rolling_mean = cumulative.rolling(window=ROLLING_WINDOW, min_periods=1).mean()
    return rolling_mean

# ============ 主流程 ============
print("=" * 60)
print("Loading CrudeBERT results...")
df_crude = load_all("Crudebert")
print(f"  Total: {len(df_crude)} rows\n")

print("Loading FinBERT (oil_news) results...")
df_oil = load_all("oil_news")
print(f"  Total: {len(df_oil)} rows\n")

print("Computing cumulative sentiment scores...")
cum_crude = compute_cumulative_sentiment(df_crude)
cum_oil = compute_cumulative_sentiment(df_oil)
print(f"  CrudeBERT: {len(cum_crude)} daily points")
print(f"  FinBERT:   {len(cum_oil)} daily points\n")

# ============ 图1: 累计情感得分（论文风格） ============
plot_name = "Cumulative Sentiment Scores — CrudeBERT vs FinBERT (2017–2024)"

fig, ax = plt.subplots(figsize=(20, 9), dpi=150)
ax.set_title(plot_name, fontsize=FONT_SIZE_LABEL, weight='bold', pad=15)

# FinBERT 红线
ax.plot(cum_oil.index, cum_oil.values,
        color='red', linewidth=1.5, label='Cum. FinBERT')

# CrudeBERT 绿线
ax.plot(cum_crude.index, cum_crude.values,
        color='tab:green', linewidth=1.5, label='Cum. CrudeBERT')

# 坐标轴
ax.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
ax.set_ylabel('Scaled Value', fontsize=FONT_SIZE_LABEL)
ax.tick_params(axis='both', labelsize=FONT_SIZE_TICK)

# X轴日期格式
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
fig.autofmt_xdate(rotation=0, ha='center')

# 图例在底部居中
ax.legend(fontsize=FONT_SIZE_TICK, loc='upper center',
          bbox_to_anchor=(0.5, -0.12), ncol=2,
          borderaxespad=0, frameon=True)

ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
fname = os.path.join(OUTPUT_DIR, "cumulative_sentiment_comparison.png")
plt.savefig(fname, dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.basename(fname)}")

# ============ 图2: 日度情感差异 (CrudeBERT - FinBERT) ============
common_idx = cum_crude.index.intersection(cum_oil.index)
diff = cum_crude.loc[common_idx] - cum_oil.loc[common_idx]

plot_name2 = "Difference in Cumulative Sentiment (CrudeBERT − FinBERT)"
fig2, ax2 = plt.subplots(figsize=(20, 6), dpi=150)
ax2.set_title(plot_name2, fontsize=FONT_SIZE_LABEL, weight='bold', pad=15)

ax2.fill_between(diff.index, diff.values, 0,
                 where=diff >= 0, color='tab:green', alpha=0.4, label='CrudeBERT higher')
ax2.fill_between(diff.index, diff.values, 0,
                 where=diff < 0, color='red', alpha=0.4, label='FinBERT higher')
ax2.plot(diff.index, diff.values, color='black', linewidth=0.8)
ax2.axhline(0, color='black', linewidth=1, linestyle='-')

ax2.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
ax2.set_ylabel('Scaled Difference', fontsize=FONT_SIZE_LABEL)
ax2.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
ax2.xaxis.set_major_locator(mdates.YearLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax2.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
fig2.autofmt_xdate(rotation=0, ha='center')
ax2.legend(fontsize=FONT_SIZE_TICK, loc='upper center',
           bbox_to_anchor=(0.5, -0.15), ncol=2,
           borderaxespad=0, frameon=True)
ax2.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
fname2 = os.path.join(OUTPUT_DIR, "cumulative_sentiment_difference.png")
plt.savefig(fname2, dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.basename(fname2)}")

# ============ 图3: 未缩放的原始累计得分 ============
raw_crude = compute_raw_cumulative(df_crude)
raw_oil = compute_raw_cumulative(df_oil)

plot_name3 = "Raw Cumulative Sentiment Scores — CrudeBERT vs FinBERT (2017–2024)"
fig3, ax3 = plt.subplots(figsize=(20, 9), dpi=150)
ax3.set_title(plot_name3, fontsize=FONT_SIZE_LABEL, weight='bold', pad=15)

ax3.plot(raw_oil.index, raw_oil.values,
         color='red', linewidth=1.5, label='Cum. FinBERT')
ax3.plot(raw_crude.index, raw_crude.values,
         color='tab:green', linewidth=1.5, label='Cum. CrudeBERT')

ax3.set_xlabel('Date', fontsize=FONT_SIZE_LABEL)
ax3.set_ylabel('Cumulative Score', fontsize=FONT_SIZE_LABEL)
ax3.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
ax3.xaxis.set_major_locator(mdates.YearLocator())
ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax3.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
fig3.autofmt_xdate(rotation=0, ha='center')
ax3.legend(fontsize=FONT_SIZE_TICK, loc='upper center',
           bbox_to_anchor=(0.5, -0.12), ncol=2,
           borderaxespad=0, frameon=True)
ax3.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
fname3 = os.path.join(OUTPUT_DIR, "raw_cumulative_sentiment.png")
plt.savefig(fname3, dpi=300, bbox_inches="tight")
print(f"Saved: {os.path.basename(fname3)}")

print(f"\nAll charts saved to: {OUTPUT_DIR}")
plt.show()
