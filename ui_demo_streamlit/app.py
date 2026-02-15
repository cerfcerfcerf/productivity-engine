"""Streamlit demo UI for productivity-engine."""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from productivity_engine.adapters import csv_adapter, json_adapter
from productivity_engine.evaluator import compare
from productivity_engine.escalation import deadline_schedule
from productivity_engine.features import extract_features
from productivity_engine.risk_model import compute_risk, heuristic_risk, train_logistic
from productivity_engine.scheduling import best_nudge_times
from productivity_engine.simulation import simulate_adaptive, simulate_baseline


ACTIONS = ["done", "snooze", "ignore", "created"]
DEMO_DATASET = Path(__file__).resolve().parents[1] / "examples" / "sample_dataset.csv"


def _parse_events_from_path(file_path: str) -> list:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".csv":
        return csv_adapter.parse(file_path)
    if suffix == ".json":
        return json_adapter.parse(file_path)
    raise ValueError("Unsupported file type. Please use .csv or .json")


def _parse_uploaded(uploaded_file) -> list:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        temp_path = handle.name
    return _parse_events_from_path(temp_path)


def _build_summary(events: list) -> dict[str, Any]:
    action_counts = Counter(event.action for event in events)
    total = len(events)
    done_pct = (action_counts.get("done", 0) / total * 100.0) if total else 0.0
    ignore_pct = (action_counts.get("ignore", 0) / total * 100.0) if total else 0.0
    return {
        "total_events": total,
        "unique_task_ids": len({event.task_id for event in events}),
        "action_counts": {action: action_counts.get(action, 0) for action in ACTIONS},
        "done_pct": done_pct,
        "ignore_pct": ignore_pct,
    }


def _fmt_hour(hour: int) -> str:
    return f"{int(hour):02d}:00"


def _serialize_events(events: list) -> list[dict[str, Any]]:
    return [
        {
            "task_id": event.task_id,
            "timestamp": event.timestamp.isoformat(),
            "action": event.action,
            "deadline_hours": event.deadline_hours,
            "category": event.category,
        }
        for event in events
    ]


def run_engine(events: list, task: dict) -> dict[str, Any]:
    """Run all engine steps and return a UI-friendly result payload."""

    features = extract_features(events)
    model = train_logistic(events)
    risk_score = compute_risk(task=task, features=features, model=model, hour=task["now_hour"])
    risk_components = heuristic_risk(task=task, features=features, hour=task["now_hour"], return_components=True)

    if risk_score < 0.33:
        risk_level = "LOW"
    elif risk_score < 0.66:
        risk_level = "MED"
    else:
        risk_level = "HIGH"

    nudge_hours = best_nudge_times(task=task, features=features, model=model)
    escalation = deadline_schedule(task["deadline_hours"]) if task["task_type"] == "deadline" else []

    baseline = simulate_baseline(events)
    adaptive = simulate_adaptive(events)
    comparison = compare(baseline, adaptive)

    return {
        "summary": _build_summary(events),
        "features": features,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "risk_components": risk_components,
        "nudge_hours": nudge_hours,
        "escalation": escalation,
        "baseline": baseline,
        "adaptive": adaptive,
        "comparison": comparison,
    }


def main() -> None:
    import streamlit as st

    @st.cache_data
    def parse_events_from_path(file_path: str) -> list[dict[str, Any]]:
        return _serialize_events(_parse_events_from_path(file_path))

    @st.cache_data
    def parse_uploaded_bytes(name: str, content: bytes) -> list[dict[str, Any]]:
        suffix = Path(name).suffix.lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(content)
            temp_path = handle.name
        return _serialize_events(_parse_events_from_path(temp_path))

    @st.cache_data
    def cached_extract_features(events_payload: list[dict[str, Any]]) -> dict[str, Any]:
        from productivity_engine.schema import NormalizedEvent
        from datetime import datetime

        events = [
            NormalizedEvent(
                task_id=e["task_id"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                action=e["action"],
                deadline_hours=e["deadline_hours"],
                category=e["category"],
            )
            for e in events_payload
        ]
        return extract_features(events)

    @st.cache_resource
    def cached_train_logistic(events_payload: list[dict[str, Any]]):
        from productivity_engine.schema import NormalizedEvent
        from datetime import datetime

        events = [
            NormalizedEvent(
                task_id=e["task_id"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                action=e["action"],
                deadline_hours=e["deadline_hours"],
                category=e["category"],
            )
            for e in events_payload
        ]
        return train_logistic(events)

    def hydrate(events_payload: list[dict[str, Any]]):
        from productivity_engine.schema import NormalizedEvent
        from datetime import datetime

        return [
            NormalizedEvent(
                task_id=e["task_id"],
                timestamp=datetime.fromisoformat(e["timestamp"]),
                action=e["action"],
                deadline_hours=e["deadline_hours"],
                category=e["category"],
            )
            for e in events_payload
        ]

    st.set_page_config(page_title="Productivity Engine Demo", layout="wide")
    st.title("Productivity Engine — Streamlit Demo")

    with st.sidebar:
        st.header("Controls")
        uploaded = st.file_uploader("Upload event log", type=["csv", "json"])
        use_demo = st.checkbox("Load demo dataset", value=True)
        task_type = st.selectbox("Task type", options=["habit", "deadline"], index=0)
        category = st.text_input("Category", value="study")
        priority = st.number_input("Priority", min_value=1, max_value=3, value=2, step=1)
        now_hour = st.slider("Now hour", min_value=0, max_value=23, value=19)
        window_start = st.slider("Window start", min_value=0, max_value=23, value=18)
        window_end = st.slider("Window end", min_value=0, max_value=23, value=21)
        if window_end < window_start:
            st.warning("Window end was before start; it will be adjusted to match start.")
            window_end = window_start

        deadline_hours = st.number_input(
            "Deadline hours",
            min_value=0.0,
            max_value=240.0,
            value=24.0,
            step=1.0,
            disabled=(task_type != "deadline"),
        )
        run = st.button("Run engine", type="primary")

    if not run:
        st.info("Configure inputs in the sidebar and click **Run engine**.")
        return

    try:
        if use_demo:
            events_payload = parse_events_from_path(str(DEMO_DATASET))
            data_source = f"demo dataset ({DEMO_DATASET})"
        elif uploaded is not None:
            events_payload = parse_uploaded_bytes(uploaded.name, uploaded.getvalue())
            data_source = f"uploaded file ({uploaded.name})"
        else:
            st.error("Please upload a CSV/JSON file or enable 'Load demo dataset'.")
            return

        if not events_payload:
            st.error("No events were found in the selected input.")
            return

        events = hydrate(events_payload)
        features = cached_extract_features(events_payload)
        model = cached_train_logistic(events_payload)

        task = {
            "task_id": "ui_demo_task",
            "task_type": task_type,
            "category": category or None,
            "priority": int(priority),
            "deadline_hours": float(deadline_hours) if task_type == "deadline" else None,
            "window": (int(window_start), int(window_end)),
            "weekday": 0,
            "now_hour": int(now_hour),
        }

        risk_score = compute_risk(task=task, features=features, model=model, hour=task["now_hour"])
        risk_components = heuristic_risk(task=task, features=features, hour=task["now_hour"], return_components=True)
        result = run_engine(events, task)
        result["risk_score"] = risk_score
        result["risk_components"] = risk_components
        result["features"] = features

        st.success(f"Loaded {len(events)} events from {data_source}.")

        st.subheader("A) Data Summary")
        summary = result["summary"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total events", summary["total_events"])
        c2.metric("Unique task_ids", summary["unique_task_ids"])
        c3.metric("% done", f"{summary['done_pct']:.2f}%")
        c4.metric("% ignore", f"{summary['ignore_pct']:.2f}%")
        st.table([summary["action_counts"]])
        st.bar_chart(summary["action_counts"])

        st.subheader("Behavior Over Hours")
        chart_data = {
            "completion_by_hour": [result["features"].get("completion_by_hour", {}).get(h, 0.0) for h in range(24)],
            "ignore_by_hour": [result["features"].get("ignore_by_hour", {}).get(h, 0.0) for h in range(24)],
        }
        st.line_chart(chart_data)

        st.subheader("B) Risk Result")
        r1, r2 = st.columns(2)
        r1.metric("risk_score", f"{result['risk_score']:.4f}")
        r2.metric("risk_level", result["risk_level"])

        st.write("**Risk factors**")
        st.table(
            [
                {
                    "hour_penalty": f"{result['risk_components']['hour_penalty']:.4f}",
                    "ignore_penalty": f"{result['risk_components']['ignore_penalty']:.4f}",
                    "deadline_urgency": f"{result['risk_components']['deadline_urgency']:.4f}",
                    "streak_bonus": f"{result['risk_components']['streak_bonus']:.4f}",
                    "base_rate": f"{result['risk_components']['base_rate']:.4f}",
                }
            ]
        )

        st.subheader("C) Recommended Schedule")
        display_times = [_fmt_hour(hour) for hour in result["nudge_hours"]]
        st.write(", ".join(display_times) if display_times else "No recommendation available.")

        if task_type == "deadline":
            st.subheader("D) Deadline Escalation")
            st.write(result["escalation"])

        st.subheader("E) Evaluation")
        ec1, ec2, ec3 = st.columns(3)
        ec1.write("**Baseline metrics**")
        ec1.table([result["baseline"]])
        ec2.write("**Adaptive metrics**")
        ec2.table([result["adaptive"]])
        ec3.write("**Comparison metrics**")
        ec3.table([result["comparison"]])

        st.download_button(
            "Download result JSON",
            data=json.dumps(result, indent=2, default=str),
            file_name="productivity_engine_result.json",
            mime="application/json",
        )

    except ValueError as exc:
        st.error(f"Input error: {exc}")
    except Exception:
        st.error("Something went wrong while running the demo. Please verify the input format.")


if __name__ == "__main__":
    main()
