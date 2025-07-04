# app.py

import os
import io
import tempfile
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
import instaloader
import requests
from moviepy.editor import VideoFileClip
from langdetect import detect

# ---- ENV / Secrets
if os.path.exists('.env'):
    load_dotenv('.env')

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", None))
if not OPENAI_API_KEY:
    st.error("No OpenAI API key found. Set in .streamlit/secrets.toml as OPENAI_API_KEY.")
    st.stop()

# --- Whisper API (using OpenAI API for simplicity, switch to WhisperX if you want alignment)
import openai
openai.api_key = OPENAI_API_KEY

# --- UI ----
st.title("Instagram Video Hook Analyzer")

st.markdown("Upload Instagram usernames (one per line) or a CSV. The app scrapes videos (100k+ likes), transcribes audio, and classifies the video hook.")

input_method = st.radio("Input method", ["Text", "CSV upload"])
if input_method == "Text":
    usernames = st.text_area("Instagram usernames (one per line)", height=100)
    usernames = [u.strip() for u in usernames.splitlines() if u.strip()]
else:
    file = st.file_uploader("Upload CSV with usernames", type=["csv"])
    if file:
        df = pd.read_csv(file)
        usernames = df.iloc[:, 0].dropna().unique().tolist()
    else:
        usernames = []

if not usernames:
    st.info("Add at least one username to begin.")
    st.stop()

st.write(f"Usernames loaded: {', '.join(usernames)}")

if "results" not in st.session_state:
    st.session_state.results = []

# --- Core logic ---
def scrape_instagram_videos(username, min_likes=100_000, max_videos=10):
    """Get recent video posts with >= min_likes for a username."""
    try:
        L = instaloader.Instaloader(
            download_video_thumbnails=False,
            save_metadata=False,
            download_comments=False,
            download_geotags=False,
            download_pictures=False,
            compress_json=False,
            quiet=True
        )
        profile = instaloader.Profile.from_username(L.context, username)
        videos = []
        for post in profile.get_posts():
            if post.is_video and post.likes >= min_likes:
                videos.append({
                    "video_url": post.video_url,
                    "likes": post.likes,
                    "shortcode": post.shortcode
                })
                if len(videos) >= max_videos:
                    break
        return videos
    except Exception as e:
        st.warning(f"Error scraping {username}: {e}")
        return []

def download_video(url):
    """Download video from URL to memory."""
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        raise Exception("Failed to download video.")

def extract_audio_from_video(video_bytes):
    """Extract audio as wav bytes using moviepy."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_video:
        tmp_video.write(video_bytes.read())
        tmp_video.flush()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
            clip = VideoFileClip(tmp_video.name)
            clip.audio.write_audiofile(tmp_audio.name, codec='pcm_s16le')
            with open(tmp_audio.name, "rb") as f:
                audio_bytes = f.read()
            clip.close()
            os.unlink(tmp_video.name)
            os.unlink(tmp_audio.name)
            return io.BytesIO(audio_bytes)

def transcribe_audio_whisper(audio_bytes):
    """Transcribe audio using OpenAI Whisper API."""
    audio_bytes.seek(0)
    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        response_format="text"
    )
    return transcript

def extract_hook_text(transcript, n_words=30):
    """Extract the hook (first n words, approximates 3–8 seconds)."""
    words = transcript.split()
    return " ".join(words[:n_words])

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def classify_hook(hook_text):
    """Simple LLM prompt-based classifier."""
    system_prompt = "You are an expert at classifying video hooks for viral social content. Given a hook, classify it as: Question, Bold Claim, Problem-Solution, Data Drop, Intrigue, Promise, Other. Return just the category."
    user_prompt = f'Hook: "{hook_text}"\nCategory:'
    try:
        resp = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=4,
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return "Unknown"

# --- Main button and pipeline
if st.button("Run Analysis"):
    results = []
    progress = st.progress(0, text="Starting...")

    total = len(usernames)
    for idx, username in enumerate(usernames):
        st.info(f"Processing {username}...")
        videos = scrape_instagram_videos(username)
        for vid in videos:
            st.write(f"Transcribing video with {vid['likes']} likes...")

            try:
                video_bytes = download_video(vid['video_url'])
                audio_bytes = extract_audio_from_video(video_bytes)
                transcript = transcribe_audio_whisper(audio_bytes)
                language = detect_language(transcript)
                if language != "en":
                    st.write(f"Skipped non-English video: {vid['video_url']}")
                    continue
                hook_text = extract_hook_text(transcript)
                hook_pattern = classify_hook(hook_text)
                results.append({
                    "Username": username,
                    "Video URL": vid['video_url'],
                    "Likes": vid['likes'],
                    "Transcript": transcript,
                    "Hook Text": hook_text,
                    "Hook Pattern": hook_pattern,
                    "View Video": f"[View Video](https://instagram.com/p/{vid['shortcode']})"
                })
            except Exception as e:
                st.warning(f"Error processing video: {e}")
                continue

        progress.progress((idx + 1) / total, text=f"Done {idx + 1}/{total}")

    if results:
        df = pd.DataFrame(results)
        st.session_state.results = df
        st.success("Done! Preview below.")
    else:
        st.warning("No qualifying videos found.")

# --- Results table
if st.session_state.get("results", None) is not None and not st.session_state.results.empty:
    st.dataframe(
        st.session_state.results[["Username", "Likes", "Hook Text", "Hook Pattern", "Transcript", "Video URL", "View Video"]],
        use_container_width=True
    )

    # --- Export buttons
    csv = st.session_state.results.to_csv(index=False)
    json = st.session_state.results.to_json(orient="records")

    st.download_button("Download CSV", csv, file_name="results.csv")
    st.download_button("Download JSON", json, file_name="results.json")
