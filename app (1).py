import streamlit as st
import plotly.express as px
import tempfile
import os
import sys

sys.path.append("/content/meeting_ai")

from transcribe import transcribe_audio, get_speaker_lines
from analyse import analyse_meeting
from sentiment import analyse_sentiment, get_overall_sentiment

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🎙️",
    layout="wide"
)

# ---- HEADER ----
st.title("🎙️ AI Meeting Intelligence System")
st.caption("Upload a meeting recording → get instant summary, action items, speaker sentiment, and searchable transcript.")

st.divider()

# ---- SIDEBAR — API KEY INPUT ----
with st.sidebar:
    st.header("⚙️ Setup")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="paste your free Groq key here",
        help="Get your free key at console.groq.com"
    )
    st.caption("Your key is never stored or sent anywhere except Groq's API.")

    st.divider()
    st.markdown("**How it works**")
    st.markdown("""
1. Upload a meeting audio file
2. Whisper transcribes it locally
3. Llama 3.1 analyses the transcript
4. TextBlob scores speaker sentiment
5. Dashboard shows everything
""")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader(
    "📂 Upload your meeting audio",
    type=["mp3", "wav", "mp4", "m4a"],
    help="Works best on recordings under 30 minutes. Clear audio gives better results."
)

if uploaded_file and not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar to continue.")

elif uploaded_file and groq_api_key:

    # save uploaded file to temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- STAGE 1: TRANSCRIPTION ----
    with st.spinner("🎙️ Transcribing audio with Whisper... (1–2 mins for a 10 min recording)"):
        try:
            transcript, segments = transcribe_audio(tmp_path)
            speaker_lines = get_speaker_lines(segments)
            st.success(f"✅ Transcription done — {len(speaker_lines)} segments detected")
        except Exception as e:
            st.error(f"❌ Transcription failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    # ---- STAGE 2: LLM ANALYSIS ----
    with st.spinner("🧠 Analysing meeting with Llama 3.1 70B..."):
        try:
            analysis = analyse_meeting(transcript, groq_api_key)
            st.success("✅ Meeting analysis complete")
        except Exception as e:
            st.error(f"❌ Analysis failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    # ---- STAGE 3: SENTIMENT ----
    with st.spinner("💬 Scoring speaker sentiment..."):
        try:
            sentiment = analyse_sentiment(speaker_lines)
            overall = get_overall_sentiment(sentiment)
            st.success("✅ Sentiment analysis complete")
        except Exception as e:
            st.error(f"❌ Sentiment failed: {e}")

    os.unlink(tmp_path)  # clean up temp file

    st.divider()

    # ---- DISPLAY: ROW 1 — summary + mood ----
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        st.subheader("📋 Meeting Summary")
        st.info(analysis.get("summary", "No summary generated"))

    with col2:
        st.subheader("Overall Mood")
        mood = overall
        st.metric(
            label=f"{mood['emoji']} Sentiment",
            value=mood["label"],
            delta=f"score: {mood['score']}"
        )

    with col3:
        st.subheader("Quick Stats")
        st.metric("Segments", len(speaker_lines))
        st.metric("Speakers", len(sentiment))
        st.metric("Topics", len(analysis.get("topics_discussed", [])))

    st.divider()

    # ---- DISPLAY: ROW 2 — action items + decisions ----
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("✅ Action Items")
        action_items = analysis.get("action_items", [])
        if action_items:
            for i, item in enumerate(action_items, 1):
                with st.container():
                    st.markdown(f"**{i}. {item.get('task', '')}**")
                    st.caption(
                        f"👤 Owner: {item.get('owner', 'TBD')}  |  "
                        f"📅 Deadline: {item.get('deadline', 'Not specified')}"
                    )
                    st.divider()
        else:
            st.caption("No action items detected in this meeting.")

    with col5:
        st.subheader("🔹 Key Decisions")
        decisions = analysis.get("key_decisions", [])
        if decisions:
            for d in decisions:
                st.markdown(f"- {d}")
        else:
            st.caption("No key decisions detected.")

        st.subheader("📌 Topics Discussed")
        topics = analysis.get("topics_discussed", [])
        if topics:
            for t in topics:
                st.markdown(f"- {t}")

    st.divider()

    # ---- DISPLAY: ROW 3 — sentiment chart ----
    st.subheader("😊 Speaker Sentiment Analysis")

    if sentiment:
        chart_data = {
            "Speaker": list(sentiment.keys()),
            "Sentiment Score": [v["score"] for v in sentiment.values()],
            "Tone": [v["label"] for v in sentiment.values()]
        }

        color_map = {
            "Positive": "#1D9E75",
            "Neutral":  "#888780",
            "Negative": "#E24B4A"
        }

        fig = px.bar(
            chart_data,
            x="Speaker",
            y="Sentiment Score",
            color="Tone",
            color_discrete_map=color_map,
            range_y=[-1, 1],
            title="Sentiment score per speaker  (–1 = very negative  |  0 = neutral  |  +1 = very positive)",
            text="Sentiment Score"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)

        # per-speaker breakdown cards
        scols = st.columns(len(sentiment))
        for i, (speaker, data) in enumerate(sentiment.items()):
            with scols[i]:
                st.metric(
                    label=f"{data['emoji']} {speaker}",
                    value=data["label"],
                    delta=f"score: {data['score']}"
                )

    st.divider()

    # ---- DISPLAY: ROW 4 — searchable transcript ----
    st.subheader("📝 Full Transcript")
    search = st.text_input("🔍 Search transcript", placeholder="Type any keyword to filter...")

    if search:
        filtered_lines = [
            line for line in speaker_lines
            if search.lower() in line["text"].lower()
        ]
        st.caption(f"Found {len(filtered_lines)} matching segments for '{search}'")
        for line in filtered_lines:
            st.markdown(f"**{line['speaker']}** `[{line['start']}s]`: {line['text']}")
    else:
        for line in speaker_lines:
            st.markdown(f"**{line['speaker']}** `[{line['start']}s]`: {line['text']}")

else:
    # empty state
    st.markdown("""
    ### How to get started
    1. Enter your free **Groq API key** in the sidebar → get it at [console.groq.com](https://console.groq.com)
    2. Upload any meeting audio file (MP3, WAV, MP4, M4A)
    3. Wait 1–2 minutes for AI processing
    4. Get your full meeting intelligence dashboard
    """)

    st.image(
        "https://images.unsplash.com/photo-1531482615713-2afd69097998?w=800&q=60",
        caption="Turn any meeting recording into structured intelligence",
        use_column_width=True
    )
