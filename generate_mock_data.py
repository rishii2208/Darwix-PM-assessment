import csv
import os
import random
from datetime import datetime, timedelta


random.seed(42)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "mock_data")
NUM_USERS = 600
SESSION_RANGE_PER_USER = (1, 6)
FEATURES_PER_SESSION_RANGE = (1, 6)
FEEDBACK_PROBABILITY = 0.35

CHANNELS = ["Organic", "Paid Search", "Referral", "Social", "Email", "Affiliate"]
REGIONS = [
    "North America",
    "Europe",
    "Latin America",
    "Asia Pacific",
    "Middle East",
    "Africa",
]
DEVICE_TYPES = ["Desktop", "Mobile", "Tablet"]
FEATURES = [
    "Dashboard",
    "Insights",
    "Notifications",
    "Collaboration",
    "Automation",
    "Reporting",
    "Settings",
    "Integrations",
]
FEEDBACK_COMMENTS = [
    "Super intuitive flow, helped me finish tasks faster.",
    "Would love to see more customization options.",
    "Ran into a few slowdowns during peak hours.",
    "The new automation routine is a game changer.",
    "Notifications feel noisy—maybe add batching?",
    "Reporting export worked flawlessly for my client update.",
    "Collaboration tools could integrate better with Slack.",
    "App crashed once while editing settings, but recovered quickly.",
    "Mobile experience is fantastic, thanks!",
    "Great overall, but the insights could be more actionable.",
]

NOW = datetime(2025, 10, 5, 12, 0, 0)
SIGNUP_START = datetime(2023, 1, 1)


def daterange(start, end):
    delta_days = (end - start).days
    return start + timedelta(days=random.randint(0, delta_days))


def timerange(start, end):
    total_seconds = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, max(total_seconds, 1)))


def generate_users():
    users = []
    for idx in range(1, NUM_USERS + 1):
        signup_date = daterange(SIGNUP_START, NOW - timedelta(days=7))
        channel = random.choice(CHANNELS)
        region = random.choice(REGIONS)
        users.append(
            {
                "user_id": f"U{idx:05d}",
                "signup_date": signup_date.strftime("%Y-%m-%d"),
                "channel": channel,
                "region": region,
            }
        )
    return users


def generate_sessions(users):
    sessions = []
    session_lookup = {}
    session_id_counter = 1

    for user in users:
        num_sessions = random.randint(*SESSION_RANGE_PER_USER)
        user_sessions = []
        signup_date = datetime.strptime(user["signup_date"], "%Y-%m-%d")
        earliest_session_date = signup_date

        for _ in range(num_sessions):
            session_day = daterange(earliest_session_date, NOW)
            session_start = datetime.combine(session_day.date(), datetime.min.time()) + timedelta(
                hours=random.randint(6, 22), minutes=random.randint(0, 59)
            )
            duration_minutes = random.randint(5, 160)
            session_end = session_start + timedelta(minutes=duration_minutes)
            device_type = random.choice(DEVICE_TYPES)

            session_id = f"S{session_id_counter:06d}"
            session_id_counter += 1

            session_record = {
                "session_id": session_id,
                "user_id": user["user_id"],
                "start_time": session_start.strftime("%Y-%m-%d %H:%M:%S"),
                "end_time": session_end.strftime("%Y-%m-%d %H:%M:%S"),
                "device_type": device_type,
            }
            sessions.append(session_record)
            user_sessions.append(
                {
                    "session_id": session_id,
                    "start": session_start,
                    "end": session_end,
                }
            )

        session_lookup[user["user_id"]] = user_sessions

    return sessions, session_lookup


def generate_feature_usage(session_lookup):
    usage_records = []
    for user_sessions in session_lookup.values():
        for session in user_sessions:
            num_features = random.randint(*FEATURES_PER_SESSION_RANGE)
            features_used = random.sample(FEATURES, k=num_features)
            for feature in features_used:
                usage_ts = timerange(session["start"], session["end"])
                usage_records.append(
                    {
                        "session_id": session["session_id"],
                        "feature_name": feature,
                        "usage_timestamp": usage_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                )
    return usage_records


def generate_feedback(users, session_lookup):
    feedback_records = []
    feedback_id_counter = 1
    for user in users:
        user_id = user["user_id"]
        sessions = session_lookup[user_id]
        for session in sessions:
            if random.random() <= FEEDBACK_PROBABILITY:
                rating = random.randint(1, 5)
                feature = random.choice(FEATURES)
                comment = random.choice(FEEDBACK_COMMENTS)
                feedback_records.append(
                    {
                        "feedback_id": f"F{feedback_id_counter:06d}",
                        "user_id": user_id,
                        "rating": rating,
                        "feature_name": feature,
                        "comments": comment,
                        "session_id": session["session_id"],
                    }
                )
                feedback_id_counter += 1
    return feedback_records


def write_csv(filename, fieldnames, rows):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return filepath


def main():
    users = generate_users()
    sessions, session_lookup = generate_sessions(users)
    feature_usage = generate_feature_usage(session_lookup)
    feedback = generate_feedback(users, session_lookup)

    files = {
        "users.csv": ("user_id", "signup_date", "channel", "region"),
        "sessions.csv": ("session_id", "user_id", "start_time", "end_time", "device_type"),
        "feature_usage.csv": ("session_id", "feature_name", "usage_timestamp"),
        "feedback.csv": (
            "feedback_id",
            "user_id",
            "rating",
            "feature_name",
            "comments",
            "session_id",
        ),
    }

    paths = {}
    paths["users"] = write_csv("users.csv", files["users.csv"], users)
    paths["sessions"] = write_csv("sessions.csv", files["sessions.csv"], sessions)
    paths["feature_usage"] = write_csv("feature_usage.csv", files["feature_usage.csv"], feature_usage)
    paths["feedback"] = write_csv("feedback.csv", files["feedback.csv"], feedback)

    summary = {
        "users": len(users),
        "sessions": len(sessions),
        "feature_usage": len(feature_usage),
        "feedback": len(feedback),
    }

    print("Mock data generated:")
    for key, count in summary.items():
        print(f"- {key}: {count} rows -> {paths[key]}")


if __name__ == "__main__":
    main()
