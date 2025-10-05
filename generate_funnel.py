import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = os.path.dirname(__file__)
REPORT_DIR = os.path.join(BASE_DIR, "report")
TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"

# Funnel counts are illustrative, aligned with the observed 72% drop after account creation.
FUNNEL_STEPS = [
    {"stage": "Visited Landing Page", "users": 15000},
    {"stage": "Started Sign Up", "users": 9500},
    {"stage": "Account Created", "users": 5200},
    {"stage": "First Transaction", "users": 1456},
]


def build_funnel_dataframe(steps):
    df = pd.DataFrame(steps)
    df["users"] = df["users"].astype(float)
    df["overall_conversion"] = df["users"] / df.loc[0, "users"]
    df["step_conversion"] = df["users"] / df["users"].shift(1)
    df.loc[0, "step_conversion"] = 1.0
    df["step_drop"] = 1 - df["step_conversion"]
    return df


def render_funnel_chart(df: pd.DataFrame, output_path: str):
    plt.rcParams.update({"figure.figsize": (10, 6), "axes.facecolor": "#f7f7f7"})

    stages = df["stage"].tolist()
    values = df["users"].tolist()
    conversions = (df["step_conversion"] * 100).round(1)
    overall = (df["overall_conversion"] * 100).round(1)

    fig, ax = plt.subplots()
    bars = ax.barh(stages[::-1], values[::-1], color="#5b8def")
    ax.invert_yaxis()
    ax.set_xlabel("Users")
    ax.set_title("Onboarding Funnel")

    for bar, users, conv, cum_conv in zip(bars, values[::-1], conversions[::-1], overall[::-1]):
        width = bar.get_width()
        ax.text(
            width * 0.99,
            bar.get_y() + bar.get_height() / 2,
            f"{int(users):,} users\n{conv:.1f}% step | {cum_conv:.1f}% cum",
            va="center",
            ha="right",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    drop_idx = df.index[df["stage"] == "Account Created"].tolist()
    if drop_idx:
        idx = drop_idx[0]
        if idx + 1 < len(df):
            drop_pct = (1 - df.loc[idx + 1, "users"] / df.loc[idx, "users"]) * 100
            ax.annotate(
                f"{drop_pct:.0f}% drop here",
                xy=(df.loc[idx + 1, "users"], len(df) - (idx + 1) - 0.5),
                xytext=(df.loc[idx + 1, "users"] + df.loc[0, "users"] * 0.25, len(df) - (idx + 1) - 0.5),
                arrowprops=dict(arrowstyle="->", color="#d94e4e", lw=2),
                color="#d94e4e",
                fontsize=11,
                fontweight="bold",
            )

    ax.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_table(df: pd.DataFrame, output_path: str):
    df_out = df.copy()
    df_out["users"] = df_out["users"].astype(int)
    df_out["overall_conversion"] = (df_out["overall_conversion"] * 100).round(1)
    df_out["step_conversion"] = (df_out["step_conversion"] * 100).round(1)
    df_out["step_drop"] = (df_out["step_drop"] * 100).round(1)
    df_out.rename(
        columns={
            "overall_conversion": "overall_conversion_pct",
            "step_conversion": "step_conversion_pct",
            "step_drop": "step_drop_pct",
        },
        inplace=True,
    )
    df_out.to_csv(output_path, index=False)


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)
    df = build_funnel_dataframe(FUNNEL_STEPS)
    chart_path = os.path.join(REPORT_DIR, "funnel.png")
    table_path = os.path.join(REPORT_DIR, "funnel_table.csv")
    render_funnel_chart(df, chart_path)
    save_table(df, table_path)

    print("Funnel assets generated:")
    print("- Chart:", chart_path)
    print("- Table:", table_path)
    print("Generated:", datetime.now().strftime(TIMESTAMP_FMT))


if __name__ == "__main__":
    main()
