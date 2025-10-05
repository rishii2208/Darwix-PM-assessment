import os
from datetime import datetime, timezone
from math import sqrt

import pandas as pd

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "mock_data")
REPORT_DIR = os.path.join(BASE_DIR, "report")
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"


def load_datasets():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"), parse_dates=["signup_date"])
    sessions = pd.read_csv(
        os.path.join(DATA_DIR, "sessions.csv"), parse_dates=["start_time", "end_time"]
    )
    feature_usage = pd.read_csv(
        os.path.join(DATA_DIR, "feature_usage.csv"), parse_dates=["usage_timestamp"]
    )
    feedback = pd.read_csv(os.path.join(DATA_DIR, "feedback.csv"))
    return users, sessions, feature_usage, feedback


def compute_activity_metrics(sessions: pd.DataFrame):
    sessions = sessions.copy()
    sessions["date"] = sessions["start_time"].dt.date
    sessions["week"] = sessions["start_time"].dt.to_period("W-MON")

    dau = sessions.groupby("date")["user_id"].nunique().sort_index()
    wau = sessions.groupby("week")["user_id"].nunique().sort_index()

    dau_summary = {
        "days_observed": int(dau.shape[0]),
        "average": float(dau.mean()),
        "median": float(dau.median()),
        "max": int(dau.max()),
        "min": int(dau.min()),
    }

    wau_summary = {
        "weeks_observed": int(wau.shape[0]),
        "average": float(wau.mean()),
        "median": float(wau.median()),
        "max": int(wau.max()),
        "min": int(wau.min()),
    }

    return dau, wau, dau_summary, wau_summary


def compute_feature_adoption(sessions: pd.DataFrame, feature_usage: pd.DataFrame):
    session_users = sessions[["session_id", "user_id"]]
    usage_with_users = feature_usage.merge(session_users, on="session_id", how="left").dropna()

    active_user_count = sessions["user_id"].nunique()
    feature_user_counts = (
        usage_with_users.drop_duplicates(subset=["user_id", "feature_name"])
        .groupby("feature_name")["user_id"]
        .nunique()
        .sort_values(ascending=False)
    )

    overall_adopters = usage_with_users["user_id"].nunique()
    adoption_table = []
    for feature, count in feature_user_counts.items():
        adoption_table.append(
            {
                "feature_name": feature,
                "unique_users": int(count),
                "adoption_rate": count / active_user_count,
            }
        )

    overall_rate = overall_adopters / active_user_count if active_user_count else 0.0

    return adoption_table, overall_rate, active_user_count


def compute_retention(users: pd.DataFrame, sessions: pd.DataFrame):
    sessions = sessions.copy()
    sessions = sessions.merge(
        users[["user_id", "signup_date"]], on="user_id", how="left", validate="many_to_one"
    )
    sessions["session_date"] = sessions["start_time"].dt.normalize()
    sessions["day_diff"] = (sessions["session_date"] - sessions["signup_date"]).dt.days
    sessions = sessions[sessions["day_diff"] >= 0]

    day_diffs = sessions.groupby("user_id")["day_diff"].apply(set)
    base_users = int(day_diffs.shape[0])

    retention_days = [1, 7, 30]
    retention = []
    for day in retention_days:
        retained = sum(1 for diffs in day_diffs if day in diffs)
        retention.append(
            {
                "day": day,
                "retained_users": retained,
                "retention_rate": retained / base_users if base_users else 0.0,
            }
        )

    return retention, base_users


def phi_coefficient(tp: int, fn: int, fp: int, tn: int) -> float:
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0.0
    return (tp * tn - fp * fn) / denominator


def compute_feature_repeat_correlation(
    users: pd.DataFrame, sessions: pd.DataFrame, feature_usage: pd.DataFrame
):
    session_counts = sessions.groupby("user_id")["session_id"].count()
    repeat_series = (session_counts >= 2).astype(int)
    repeat_series = repeat_series.reindex(users["user_id"], fill_value=0)

    feature_users = (
        feature_usage.merge(sessions[["session_id", "user_id"]], on="session_id", how="left")
        .dropna(subset=["user_id"])
        .drop_duplicates(subset=["user_id", "feature_name"])
    )

    user_count = repeat_series.shape[0]
    repeat_total = int(repeat_series.sum())

    stats = []
    for feature, group in feature_users.groupby("feature_name"):
        users_used = set(group["user_id"].tolist())
        used_count = len(users_used)

        tp = sum(repeat_series.get(user, 0) for user in users_used)
        fp = used_count - tp
        fn = repeat_total - tp
        tn = user_count - (tp + fp + fn)

        repeat_rate_used = tp / used_count if used_count else 0.0
        not_used_count = user_count - used_count
        repeat_rate_not_used = (
            (repeat_total - tp) / not_used_count if not_used_count else 0.0
        )

        stats.append(
            {
                "feature_name": feature,
                "users_used": used_count,
                "repeat_rate_used": repeat_rate_used,
                "repeat_rate_not_used": repeat_rate_not_used,
                "repeat_rate_lift": repeat_rate_used - repeat_rate_not_used,
                "phi": phi_coefficient(tp, fn, fp, tn),
            }
        )

    stats.sort(key=lambda x: x["phi"], reverse=True)
    return stats


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_markdown(
    dau_summary,
    wau_summary,
    dau,
    wau,
    adoption_table,
    overall_rate,
    active_user_count,
    retention,
    retention_base,
    feature_repeat_stats,
):
    lines = []
    lines.append("# Product Metrics & Behavioral Analysis\n")
    lines.append(
        f"Generated on {datetime.now(timezone.utc).strftime(TIMESTAMP_FORMAT)} UTC\n"
    )

    lines.append("## Metric Definitions & Results\n")
    lines.append("- **Daily Active Users (DAU):** Unique users with at least one session on a given day.\n")
    lines.append(
        f"  Average DAU: **{dau_summary['average']:.1f}** across {dau_summary['days_observed']} days "
        f"(median {dau_summary['median']:.1f}, min {dau_summary['min']}, max {dau_summary['max']}).\n"
    )
    lines.append("- **Weekly Active Users (WAU):** Unique users with at least one session during an ISO week (Monday start).\n")
    lines.append(
        f"  Average WAU: **{wau_summary['average']:.1f}** across {wau_summary['weeks_observed']} weeks "
        f"(median {wau_summary['median']:.1f}, min {wau_summary['min']}, max {wau_summary['max']}).\n"
    )
    lines.append(
        "- **Feature Adoption Rate:** Share of active users who engaged with each feature at least once during the observed window.\n"
    )
    lines.append(
        f"  Overall feature adoption: **{format_percentage(overall_rate)}** of {active_user_count} active users touched any feature.\n"
    )
    lines.append(
        "- **Retention Rate (Day 1 / 7 / 30):** Percentage of new users with a session exactly N days after signup.\n"
    )
    lines.append(
        f"  Cohort size: **{retention_base}** users with post-signup activity.\n"
    )

    lines.append("\n### Daily Active Users (last 7 days of data)\n")
    dau_tail = dau.tail(7)
    if dau_tail.empty:
        lines.append("No daily activity recorded.\n")
    else:
        lines.append("| Date | Active Users |\n| --- | ---: |\n")
        for date, count in dau_tail.items():
            lines.append(f"| {date} | {count} |\n")

    lines.append("\n### Weekly Active Users (last 6 weeks of data)\n")
    wau_tail = wau.tail(6)
    if wau_tail.empty:
        lines.append("No weekly activity recorded.\n")
    else:
        lines.append("| Week (Mon start) | Active Users |\n| --- | ---: |\n")
        for week, count in wau_tail.items():
            lines.append(f"| {week} | {count} |\n")

    lines.append("\n### Feature Adoption Detail\n")
    lines.append("| Feature | Users | Adoption Rate |\n| --- | ---: | ---: |")
    for row in adoption_table:
        lines.append(
            f"| {row['feature_name']} | {row['unique_users']} | {format_percentage(row['adoption_rate'])} |"
        )

    lines.append("\n### Retention Summary\n")
    lines.append("| Day | Retained Users | Retention Rate |\n| ---: | ---: | ---: |")
    for row in retention:
        lines.append(
            f"| {row['day']} | {row['retained_users']} | {format_percentage(row['retention_rate'])} |"
        )

    lines.append("\n### Features Correlated with Repeat Sessions\n")
    lines.append(
        "Repeat sessions defined as users with 2+ sessions in the period. Phi coefficients capture the strength of\n"
        "association between touching a feature and being a repeat user (higher = stronger positive correlation).\n"
    )
    lines.append(
        "| Feature | Users | Repeat Rate (Used) | Repeat Rate (Not Used) | Lift | Phi |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    for row in feature_repeat_stats:
        lines.append(
            "| {feature} | {users} | {rru} | {rrn} | {lift} | {phi} |".format(
                feature=row["feature_name"],
                users=row["users_used"],
                rru=format_percentage(row["repeat_rate_used"]),
                rrn=format_percentage(row["repeat_rate_not_used"]),
                lift=format_percentage(row["repeat_rate_lift"]),
                phi=f"{row['phi']:.3f}",
            )
        )

    return "\n".join(lines) + "\n"


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    users, sessions, feature_usage, feedback = load_datasets()

    dau, wau, dau_summary, wau_summary = compute_activity_metrics(sessions)
    adoption_table, overall_rate, active_user_count = compute_feature_adoption(sessions, feature_usage)
    retention, retention_base = compute_retention(users, sessions)
    feature_repeat_stats = compute_feature_repeat_correlation(users, sessions, feature_usage)

    markdown = render_markdown(
        dau_summary,
        wau_summary,
        dau,
        wau,
        adoption_table,
        overall_rate,
        active_user_count,
        retention,
        retention_base,
        feature_repeat_stats,
    )

    output_path = os.path.join(REPORT_DIR, "metrics_summary.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("Metrics summary generated at:", output_path)


if __name__ == "__main__":
    main()
