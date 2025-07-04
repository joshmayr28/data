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

# Load API key from Streamlit secrets or .env for local dev
if os.path.exists('.env'):
    load_dotenv('.env')
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", None))
if not OPENAI_API_KEY:
    st.error("No OpenAI API key found. Add to .streamlit/secrets.toml as OPENAI_API_KEY.")
    st.stop()

import openai
openai.api_key = OPENAI_API_KEY

st.title("Instagram Video Hook Analyzer")

st.markdown("Upload Instagram usernames or CSV, set your filters, and analyze high-performing IG videos.")

# ---- Inputs ----
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

st.subheader("Custom Filters")
min_followers = st.number_input("Minimum Followers", min_value=0, value=0)
min_likes = st.number_input("Minimum Likes (per video)", min_value=0, value=100000)
min_views = st.number_input("Minimum Video Views (per video)", min_value=0, value=0)

if not usernames:
    st.info("Add at least one username to begin.")
    st.stop()

if "results" not in st.session_state:
    st.session_state.results = []

# ---- Scrape logic ----
def scrape_instagram_videos(username, min_followers=0, min_likes=100_000, min_views=0, max_videos=10):
    """Scrape IG video posts with filtering on followers, likes, views."""
    try:
        L = instaloader.Instaloader(quiet=True)
        profile = instaloader.Profile.from_username(L.context, username)
        # Profile filter
        if profile.followers < min_followers:
            st.info(f"Skipping {username}: only {profile.followers} followers (min {min_followers})")
            return [], profile.followers
        videos = []
        for post in profile.get_posts():
            if not post.is_video:
                continue
            if post.likes < min_likes:
                continue
            views = getattr(post, 'video_view_count', 0) or 0
            if views < min_views:
                continue
            videos.append({
                "video_url": post.video_url,
                "likes": post.likes,
                "views": views,
                "shortcode": post.shortcode
            })
            if len(videos) >= max_videos:
                break
        return videos, profile.followers
    except Exception as e:
        st.warning(f"Error scraping {username}: {e}")
        return [], None

def download_video(url):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        return io.BytesIO(response.content)
    else:
        raise Exception("Failed to download video.")

def extract_audio_from_video(video_bytes):
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
    audio_bytes.seek(0)
    transcript = openai.audio.transcriptions.create(
        model="whisper-1",
        file=audio_bytes,
        response_format="text"
    )
    return transcript

def extract_hook_text(transcript, n_words=30):
    words = transcript.split()
    return " ".join(words[:n_words])

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def classify_hook(hook_text):
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
    except Exception:
        return "Unknown"

# ---- Main pipeline ----
if st.button("Run Analysis"):
    results = []
    progress = st.progress(0, text="Starting...")
    total = len(usernames)
    for idx, username in enumerate(usernames):
        st.info(f"Processing {username}...")
        videos, followers = scrape_instagram_videos(
            username,
            min_followers=min_followers,
            min_likes=min_likes,
            min_views=min_views
        )
        for vid in videos:
            try:
                st.write(f"Transcribing video ({vid['likes']} likes, {vid['views']} views)...")
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
                    "Followers": followers,
                    "Video URL": vid['video_url'],
                    "Likes": vid['likes'],
                    "Views": vid['views'],
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

# ---- Results table and download ----
if st.session_state.get("results", None) is not None and not st.session_state.results.empty:
    st.dataframe(
        st.session_state.results[
            ["Username", "Followers", "Likes", "Views", "Hook Text", "Hook Pattern", "Transcript", "Video URL", "View Video"]
        ],
        use_container_width=True
    )
    csv = st.session_state.results.to_csv(index=False)
    json = st.session_state.results.to_json(orient="records")
    st.download_button("Download CSV", csv, file_name="results.csv")
    st.download_button("Download JSON", json, file_name="results.json")
