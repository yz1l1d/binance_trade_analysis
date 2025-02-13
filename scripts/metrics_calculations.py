import pandas as pd
import numpy as np


df = pd.read_csv("data\\processed_trades.csv")


df['realizedProfit'] = df['realizedProfit'].astype(float)
df['quantity'] = df['quantity'].astype(float)


grouped = df.groupby("Port_IDs")


def calculate_roi(df):
    total_pnl = df["realizedProfit"].sum()
    initial_investment = df["quantity"].sum()  
    return (total_pnl / initial_investment) * 100 if initial_investment > 0 else 0


def calculate_sharpe_ratio(df):
    risk_free_rate = 0.02  
    returns = df["realizedProfit"] / df["quantity"]
    excess_returns = returns - risk_free_rate
    return excess_returns.mean() / excess_returns.std() if excess_returns.std() > 0 else 0


def calculate_mdd(df):
    df['cumulative_pnl'] = df["realizedProfit"].cumsum()
    peak = df['cumulative_pnl'].cummax()
    drawdown = (df['cumulative_pnl'] - peak) / peak
    return drawdown.min()  


def calculate_win_rate(df):
    win_positions = (df["realizedProfit"] > 0).sum()
    total_positions = len(df)
    return (win_positions / total_positions) * 100 if total_positions > 0 else 0


metrics = grouped.apply(lambda x: pd.Series({
    "ROI": calculate_roi(x),
    "PnL": x["realizedProfit"].sum(),
    "Sharpe_Ratio": calculate_sharpe_ratio(x),
    "MDD": calculate_mdd(x),
    "Win_Rate": calculate_win_rate(x),
    "Win_Positions": (x["realizedProfit"] > 0).sum(),
    "Total_Positions": len(x)
}), include_groups=False)


metrics.to_csv("outputs/account_performance_metrics.csv")



metrics["ROI"] = (metrics["ROI"] - metrics["ROI"].min()) / (metrics["ROI"].max() - metrics["ROI"].min())
metrics["PnL"] = (metrics["PnL"] - metrics["PnL"].min()) / (metrics["PnL"].max() - metrics["PnL"].min())
metrics["Sharpe_Ratio"] = (metrics["Sharpe_Ratio"] - metrics["Sharpe_Ratio"].min()) / (metrics["Sharpe_Ratio"].max() - metrics["Sharpe_Ratio"].min())
metrics["Win_Rate"] = (metrics["Win_Rate"] - metrics["Win_Rate"].min()) / (metrics["Win_Rate"].max() - metrics["Win_Rate"].min())


weights = {
    "ROI": 0.3,
    "PnL": 0.2,
    "Sharpe_Ratio": 0.2,
    "Win_Rate": 0.2,
    "MDD": -0.1  
}


metrics["Score"] = (
    metrics["ROI"] * weights["ROI"] +
    metrics["PnL"] * weights["PnL"] +
    metrics["Sharpe_Ratio"] * weights["Sharpe_Ratio"] +
    metrics["Win_Rate"] * weights["Win_Rate"] +
    metrics["MDD"] * weights["MDD"]
)


metrics = metrics.sort_values("Score", ascending=False)


top_20 = metrics.head(20)
top_20.to_csv("outputs/top_20_accounts.csv")

print("Ranking complete! Top 20 accounts saved.")

