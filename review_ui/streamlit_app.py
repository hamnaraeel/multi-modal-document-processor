"""Human-in-the-loop review dashboard + analytics, talking to the FastAPI
ingestion/review API over HTTP so this UI can run anywhere (including as
its own Docker service) independent of the backend.
"""

from __future__ import annotations

import os
import time

import pandas as pd
import plotly.express as px
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Document Processor", layout="wide")


def confidence_color(score: float | None) -> str:
    if score is None:
        return "#9e9e9e"
    if score >= 0.85:
        return "#2e7d32"  # green
    if score >= 0.6:
        return "#f9a825"  # yellow
    return "#c62828"  # red


def confidence_badge(field: str, score: float | None) -> str:
    color = confidence_color(score)
    label = f"{score:.2f}" if score is not None else "n/a"
    return (
        f"<span style='background-color:{color}; color:white; padding:2px 8px; "
        f"border-radius:10px; font-size:0.75em; margin-left:6px;'>{field}: {label}</span>"
    )


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_upload():
    st.header("Upload a Document")
    source = st.text_input("Source / vendor identifier (optional)", "")
    uploaded = st.file_uploader(
        "PDF, JPEG, PNG, or TIFF", type=["pdf", "jpg", "jpeg", "png", "tif", "tiff"]
    )

    if uploaded and st.button("Process Document", type="primary"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        params = {"source": source} if source else {}
        response = requests.post(f"{API_BASE_URL}/documents", files=files, params=params, timeout=60)
        if response.ok:
            data = response.json()
            st.success(f"Uploaded. Document ID: {data['document_id']} -- status: {data['status']}")
            st.session_state["last_uploaded_id"] = data["document_id"]
        else:
            st.error(f"Upload failed: {response.text}")

    if "last_uploaded_id" in st.session_state:
        doc_id = st.session_state["last_uploaded_id"]
        if st.button("Check status"):
            resp = requests.get(f"{API_BASE_URL}/documents/{doc_id}/status", timeout=30)
            if resp.ok:
                st.json(resp.json())


def page_review_queue():
    st.header("Review Queue")
    resp = requests.get(f"{API_BASE_URL}/review/queue", timeout=30)
    if not resp.ok:
        st.error("Could not load review queue.")
        return

    items = resp.json()
    if not items:
        st.info("Nothing pending review right now.")
        return

    df = pd.DataFrame(items)
    st.dataframe(df, use_container_width=True)

    selected_id = st.selectbox(
        "Open a document for review",
        options=[i["document_id"] for i in items],
        format_func=lambda doc_id: next(i["filename"] for i in items if i["document_id"] == doc_id),
    )
    if st.button("Open"):
        st.session_state["reviewing_document_id"] = selected_id
        st.session_state["review_started_at"] = time.time()
        st.rerun()


def page_document_review():
    doc_id = st.session_state.get("reviewing_document_id")
    if not doc_id:
        st.info("Pick a document from the Review Queue page first.")
        return

    resp = requests.get(f"{API_BASE_URL}/documents/{doc_id}/detail", timeout=30)
    if not resp.ok:
        st.error("Could not load document detail.")
        return
    detail = resp.json()

    st.header(f"Reviewing: {detail['filename']}")
    st.caption(
        f"Type: {detail['document_type']} | Overall confidence: "
        f"{detail['overall_confidence']:.2f} | Routing: {detail['routing_decision']} "
        f"({detail['routing_reason']})"
    )

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Source Document")
        for url in detail["page_image_urls"]:
            st.image(f"{API_BASE_URL}{url}", use_container_width=True)

    with right:
        st.subheader("Extracted Fields")
        fields = detail["fields"] or {}
        confidences = detail["field_confidence"] or {}
        sources = detail["field_sources"] or {}

        corrections_to_submit = {}
        for field_name, value in fields.items():
            score = confidences.get(field_name)
            st.markdown(confidence_badge(field_name, score), unsafe_allow_html=True)
            new_value = st.text_input(
                label=field_name, value="" if value is None else str(value), key=f"field_{field_name}"
            )
            source_info = sources.get(field_name) or {}
            if source_info.get("snippet"):
                st.caption(f"Source: “{source_info['snippet']}” (page {source_info.get('page_numbers')})")
            if new_value != ("" if value is None else str(value)):
                corrections_to_submit[field_name] = (value, new_value)

        if detail["type_errors"]:
            st.error(f"Type validation issues: {detail['type_errors']}")
        if detail["business_rule_failures"]:
            st.error(f"Business rule failures: {detail['business_rule_failures']}")
        if detail["business_rule_warnings"]:
            st.warning(f"Business rule warnings: {detail['business_rule_warnings']}")
        if detail["anomalies"]:
            st.info(f"Anomalies: {detail['anomalies']}")
        if detail["conflicts"]:
            st.warning(f"Chunk-merge conflicts: {detail['conflicts']}")

        reviewer = st.text_input("Reviewer name", value=st.session_state.get("reviewer_name", ""))
        st.session_state["reviewer_name"] = reviewer

        col_a, col_b, col_c = st.columns(3)
        started_at = st.session_state.get("review_started_at", time.time())

        def finalize(decision: str):
            for field_name, (original, corrected) in corrections_to_submit.items():
                requests.post(
                    f"{API_BASE_URL}/documents/{doc_id}/corrections",
                    json={
                        "field_name": field_name,
                        "original_value": None if original is None else str(original),
                        "corrected_value": corrected,
                        "reviewer": reviewer or "unknown",
                        "correction_type": "extraction_error",
                    },
                    timeout=30,
                )
            requests.post(
                f"{API_BASE_URL}/documents/{doc_id}/review",
                json={
                    "reviewer": reviewer or "unknown",
                    "decision": decision,
                    "review_seconds": time.time() - started_at,
                },
                timeout=30,
            )
            st.session_state.pop("reviewing_document_id", None)
            st.success(f"Recorded decision: {decision}")
            st.rerun()

        with col_a:
            if st.button("Approve", type="primary"):
                finalize("edited" if corrections_to_submit else "approved")
        with col_b:
            if st.button("Reject"):
                finalize("rejected")
        with col_c:
            st.caption("Keyboard shortcuts: use Tab/Enter to move fast through fields.")


def page_analytics():
    st.header("Processing Analytics")

    volume = requests.get(f"{API_BASE_URL}/analytics/volume", timeout=30).json()
    approval = requests.get(f"{API_BASE_URL}/analytics/auto-approval-rate", timeout=30).json()
    accuracy = requests.get(f"{API_BASE_URL}/analytics/extraction-accuracy", timeout=30).json()
    review_time = requests.get(f"{API_BASE_URL}/analytics/review-time", timeout=30).json()
    ocr = requests.get(f"{API_BASE_URL}/analytics/ocr-engines", timeout=30).json()

    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Auto-Approval Rate", f"{approval['overall'] * 100:.1f}%")
    col2.metric("Avg Review Time (s)", review_time.get("overall_avg_seconds") or "n/a")
    col3.metric("Vision Fallback Rate", f"{ocr['vision_fallback_rate'] * 100:.1f}%")

    if volume:
        st.plotly_chart(px.bar(pd.DataFrame(volume), x="date", y="count", title="Documents Processed Per Day"), use_container_width=True)

    if approval["over_time"]:
        st.plotly_chart(
            px.line(
                pd.DataFrame(approval["over_time"]), x="date", y="auto_approval_rate",
                title="Auto-Approval Rate Over Time",
            ),
            use_container_width=True,
        )

    st.subheader("OCR Engine Comparison")
    st.json(ocr)

    st.subheader("Extraction Accuracy by Field (from human corrections)")
    if accuracy:
        st.dataframe(pd.DataFrame(accuracy), use_container_width=True)
    else:
        st.info("No corrections logged yet -- accuracy tracking starts once reviewers begin editing fields.")


PAGES = {
    "Upload": page_upload,
    "Review Queue": page_review_queue,
    "Document Review": page_document_review,
    "Analytics": page_analytics,
}


def main():
    st.sidebar.title("Document Processor")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    PAGES[choice]()


if __name__ == "__main__":
    main()
