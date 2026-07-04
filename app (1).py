import streamlit as st
import plotly.express as px
import tempfile
import os
import sys

sys.path.append("/content/meeting_ai")

from chunker import process_any_file
from transcribe import get_speaker_lines
from analyse import analyse_meeting
from sentiment import analyse_sentiment, get_overall_sentiment

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="AI Meeting Intelligence",
    page_icon="🎙️",
    layout="wide"
)

# ---- READ GROQ KEY ----
try:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
    key_source = "secrets"
except Exception:
    GROQ_API_KEY = None
    key_source = "sidebar"

# ---- HEADER ----
st.title("🎙️ AI Meeting Intelligence System")
st.caption("Upload any meeting audio or video → AI automatically splits, transcribes, and analyses it.")
st.divider()


# ---- SIDEBAR ----
with st.sidebar:
    st.header("⚙️ Setup")

    # always show the input box
    try:
        default_key = st.secrets["GROQ_API_KEY"]
        demo_mode = True
    except Exception:
        default_key = ""
        demo_mode = False

    if demo_mode:
        st.success("✅ Demo mode — API key pre-loaded")
        st.caption("You can also use your own free key below")

    user_key = st.text_input(
        "Groq API Key",
        value="" if demo_mode else "",
        type="password",
        placeholder="paste your free key here (optional in demo)",
        help="Get your free key at console.groq.com — no credit card needed"
    )

    if user_key.strip():
        GROQ_API_KEY = user_key.strip()
        st.success("✅ Using your personal API key")
    elif demo_mode:
        GROQ_API_KEY = default_key
        st.info("ℹ️ Using demo API key")
    else:
        GROQ_API_KEY = None
        st.warning("⚠️ Please enter a Groq API key")

    st.divider()

    # ← THIS BLOCK WAS MISSING — add it back
    st.header("🔧 Settings")
    chunk_minutes = st.slider(
        "Chunk size (minutes)",
        min_value=2,
        max_value=10,
        value=5,
        help="Long audio is split into chunks of this size."
    )

    st.divider()
    st.markdown("**🆓 Get your free key**")
    st.markdown("[console.groq.com](https://console.groq.com) → Sign up → API Keys → Create")
    st.caption("Free. No credit card. 14,400 requests/day.")

    st.divider()
    st.markdown("**✨ New features**")
    st.markdown("""
- 🎬 Supports VIDEO files (MP4, MOV)
- ✂️ Auto-splits long audio into chunks
- 🔄 Processes each chunk separately
- 📝 Joins everything into one transcript
- No manual cutting needed!
""")
    st.markdown("**Supported formats**")
    st.markdown("""
**Audio:** MP3, WAV, M4A\n
**Video:** MP4, MOV, AVI\n
**Any length:** 5 min → 3 hours
""")

# ---- FILE UPLOAD ----
uploaded_file = st.file_uploader(
    "📂 Upload meeting audio or video (any length)",
    type=["mp3", "wav", "mp4", "m4a", "mov", "avi"],
    help="Any length supported — app auto-splits into chunks"
)

if uploaded_file and not GROQ_API_KEY:
    st.warning("⚠️ Please enter your Groq API key in the sidebar.")

elif uploaded_file and GROQ_API_KEY:

    # show file info
    file_size_mb = uploaded_file.size / (1024 * 1024)
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    is_video = file_ext in [".mp4", ".mov", ".avi", ".mkv"]

    col_info1, col_info2, col_info3 = st.columns(3)
    with col_info1:
        st.metric("File", uploaded_file.name[:25])
    with col_info2:
        st.metric("Size", f"{round(file_size_mb, 1)} MB")
    with col_info3:
        st.metric("Type", "🎬 Video" if is_video else "🎙️ Audio")

    # save uploaded file temporarily
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    # ---- STAGE 1: AUTO PROCESS (convert + split + transcribe) ----
    with st.spinner(f"{'🎬 Converting video + ' if is_video else ''}✂️ Auto-splitting + 🎙️ Transcribing all chunks... (may take a few minutes for long audio)"):
        try:
            # progress info
            if is_video:
                st.info("🎬 Video detected — converting to audio first, then splitting into chunks...")
            else:
                st.info(f"✂️ Audio will be automatically split into {chunk_minutes} minute chunks and transcribed...")

            transcript, segments = process_any_file(tmp_path, chunk_minutes=chunk_minutes)
            speaker_lines = get_speaker_lines(segments)
            st.success(f"✅ Transcription complete — {len(speaker_lines)} segments from full audio")

        except Exception as e:
            st.error(f"❌ Processing failed: {e}")
            os.unlink(tmp_path)
            st.stop()

    # ---- STAGE 2: LLM ANALYSIS ----
    with st.spinner("🧠 Analysing full transcript with Llama 3.3 70B..."):
        try:
            analysis = analyse_meeting(transcript, GROQ_API_KEY)
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

    os.unlink(tmp_path)
    st.divider()

    # ---- ROW 1: SUMMARY + STATS ----
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
        st.metric("Total Segments", len(speaker_lines))
        st.metric("Speakers", len(sentiment))
        st.metric("Topics Found", len(analysis.get("topics_discussed", [])))

    st.divider()

    # ---- ROW 2: ACTION ITEMS + DECISIONS ----
    col4, col5 = st.columns(2)

    with col4:
        st.subheader("✅ Action Items")
        action_items = analysis.get("action_items", [])
        if action_items:
            for i, item in enumerate(action_items, 1):
                st.markdown(f"**{i}. {item.get('task', '')}**")
                st.caption(
                    f"👤 Owner: {item.get('owner', 'TBD')}  |  "
                    f"📅 Deadline: {item.get('deadline', 'Not specified')}"
                )
                st.divider()
        else:
            st.caption("No action items detected.")

    with col5:
        st.subheader("🔹 Key Decisions")
        for d in analysis.get("key_decisions", []):
            st.markdown(f"- {d}")

        st.subheader("📌 Topics Discussed")
        for t in analysis.get("topics_discussed", []):
            st.markdown(f"- {t}")

    st.divider()

    # ---- ROW 3: SENTIMENT CHART ----
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
            title="Sentiment score per speaker  (–1 = negative  |  0 = neutral  |  +1 = positive)",
            text="Sentiment Score"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True)

        scols = st.columns(len(sentiment))
        for i, (speaker, data) in enumerate(sentiment.items()):
            with scols[i]:
                st.metric(
                    label=f"{data['emoji']} {speaker}",
                    value=data["label"],
                    delta=f"score: {data['score']}"
                )

    st.divider()

    # ---- ROW 4: SEARCHABLE TRANSCRIPT ----
    st.subheader("📝 Full Transcript")
    search = st.text_input("🔍 Search transcript", placeholder="Type any keyword...")

    if search:
        filtered = [l for l in speaker_lines if search.lower() in l["text"].lower()]
        st.caption(f"Found {len(filtered)} segments matching '{search}'")
        for line in filtered:
            st.markdown(f"**{line['speaker']}** `[{line['start']}s]`: {line['text']}")
    else:
        for line in speaker_lines:
            st.markdown(f"**{line['speaker']}** `[{line['start']}s]`: {line['text']}")

else:
    st.markdown("""
    ### 👋 Welcome! Here is how to get started:

    **Step 1** — Enter your free Groq API key in the sidebar
    > Get it free at [console.groq.com](https://console.groq.com)

    **Step 2** — Upload any meeting audio or video
    > Any length supported — 5 mins to 3 hours ✅

    **Step 3** — App automatically:
    - 🎬 Converts video to audio (if needed)
    - ✂️ Splits into 5 minute chunks
    - 🎙️ Transcribes each chunk
    - 🧠 Analyses the full transcript
    - 😊 Scores each speaker's sentiment

    **Step 4** — Get your full meeting intelligence dashboard ✨
    """)
