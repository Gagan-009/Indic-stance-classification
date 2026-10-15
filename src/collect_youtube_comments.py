"""Collect genuine Hindi and Bengali YouTube comments using the official YouTube Data API v3."""

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.prepare_dataset import detect_indic_language
from src.text_cleaner import clean_indic_text

# Curated seed pool of verified climate change videos from reputable media channels
SEED_CLIMATE_VIDEOS = [
    # --- Hindi Videos ---
    {
        "video_id": "tchrElKMiMM",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "Climate Change impact on developing nations",
    },
    {
        "video_id": "nH6gFgTpnYU",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "Severe heatwaves and climate burden",
    },
    {
        "video_id": "VxK8ETqGNfU",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "COP26 climate promises and emissions",
    },
    {
        "video_id": "ZBpD-ocIi0g",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "Extreme weather and health dangers",
    },
    {
        "video_id": "vpuw7jPg2Lk",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "Agriculture, water depletion, and climate crisis",
    },
    {
        "video_id": "MlbKTUkifLI",
        "lang": "hi",
        "channel": "BBC News Hindi",
        "topic": "Global warming discourse in India",
    },
    {
        "video_id": "0UydzLKcFww",
        "lang": "hi",
        "channel": "DW Documentary Hindi",
        "topic": "क्या जलवायु परिवर्तन सच्चाई है या झूठ?",
    },
    {
        "video_id": "BE3dPTCjt5U",
        "lang": "hi",
        "channel": "NDTV India",
        "topic": "जलवायु परिवर्तन और ग्लोबल वॉर्मिंग - जानें सब कुछ",
    },
    {
        "video_id": "JSkCLLP8AWs",
        "lang": "hi",
        "channel": "NDTV India",
        "topic": "Global Warming से Himalayan Region में पिघलते Glacier",
    },
    {
        "video_id": "Fu9J106rLgE",
        "lang": "hi",
        "channel": "DW Hindi",
        "topic": "इंसान और पृथ्वी का बुखार बिल्कुल एक जैसा है",
    },
    # --- Bengali Videos ---
    {
        "video_id": "EXP1XkAN528",
        "lang": "bn",
        "channel": "BBC News Bangla",
        "topic": "Rising global temperatures and warming atmosphere",
    },
    {
        "video_id": "nJ8JI7J7LKs",
        "lang": "bn",
        "channel": "BBC News Bangla",
        "topic": "Climate change and displaced coastal refugees",
    },
    {
        "video_id": "E0BF4nStVM4",
        "lang": "bn",
        "channel": "BBC News Bangla",
        "topic": "Carbon capture and stopping climate change",
    },
    {
        "video_id": "slzPKP5BjUE",
        "lang": "bn",
        "channel": "BBC News Bangla",
        "topic": "Future climate warming projections",
    },
    {
        "video_id": "6KzBTJftX0s",
        "lang": "bn",
        "channel": "The Business Standard",
        "topic": "Coastal life, global warming, and environmental resilience",
    },
    {
        "video_id": "ZnD1ysUh2Og",
        "lang": "bn",
        "channel": "Somoy TV",
        "topic": "জলবায়ু পরিবর্তনের প্রভাব",
    },
    {
        "video_id": "90TP5B80uRg",
        "lang": "bn",
        "channel": "Jamuna TV Plus",
        "topic": "জলবায়ু পরিবর্তনের ফলে কী কী প্রভাব পড়ছে পরিবেশে",
    },
    {
        "video_id": "G3hPfIBSG_4",
        "lang": "bn",
        "channel": "Prothom Alo",
        "topic": "জলবায়ু সংকটে নারীর ঝুঁকি বাড়ছে",
    },
    {
        "video_id": "0R-n8NKtsGc",
        "lang": "bn",
        "channel": "Jamuna TV Plus",
        "topic": "জলবায়ু পরিবর্তন সম্পর্কে সচেতনতা বৃদ্ধি জরুরি",
    },
]


def load_api_key() -> str:
    """Load and validate YouTube Data API key from environment or .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    key = os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key:
        raise ValueError(
            "YOUTUBE_API_KEY not found in environment or .env file. "
            "Please add YOUTUBE_API_KEY=your_key to .env in the project root."
        )
    return key


def fetch_comments_for_video(
    video_id: str,
    api_key: str,
    max_comments: int = 100,
    expected_lang: str = "hi",
) -> List[Dict]:
    """Fetch top-level comments for a specific YouTube video via YouTube Data API v3.

    Args:
        video_id: 11-character YouTube video ID.
        api_key: YouTube Data API v3 key.
        max_comments: Maximum number of comments to retrieve.
        expected_lang: Expected language ('hi' or 'bn').

    Returns:
        List of extracted comment dictionaries.
    """
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    page_token = None

    while len(comments) < max_comments:
        per_page = min(100, max_comments - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": per_page,
            "textFormat": "plainText",
            "key": api_key,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            resp = requests.get(url, params=params, timeout=12)
            if resp.status_code == 403:
                err_msg = resp.json().get("error", {}).get("message", "Forbidden")
                print(f"  [403 Forbidden] video {video_id}: {err_msg}")
                break
            elif resp.status_code != 200:
                print(f"  [HTTP {resp.status_code}] video {video_id}: {resp.text[:120]}")
                break

            data = resp.json()
            items = data.get("items", [])
            if not items:
                break

            for item in items:
                snippet = item["snippet"]["topLevelComment"]["snippet"]
                raw_text = snippet.get("textOriginal", "")
                cleaned = clean_indic_text(raw_text)

                # Skip completely blank or trivial single-word comments
                if len(cleaned.strip()) < 5:
                    continue

                detected = detect_indic_language(cleaned, fallback_lang="other")

                comments.append(
                    {
                        "comment_id": item["id"],
                        "video_id": video_id,
                        "author_display_name": snippet.get("authorDisplayName", "Anonymous"),
                        "published_at": snippet.get("publishedAt", ""),
                        "like_count": int(snippet.get("likeCount", 0)),
                        "raw_text": raw_text,
                        "clean_text": cleaned,
                        "detected_language": detected,
                        "target_claim": "Climate change and global warming is a serious concern.",
                        "provenance": "youtube_data_api_v3_harvest",
                    }
                )

            page_token = data.get("nextPageToken")
            if not page_token:
                break

            time.sleep(0.2)  # Respect API rate limits
        except Exception as exc:
            print(f"  [Error] Failed to fetch comments for {video_id}: {exc}")
            break

    return comments


def harvest_comments(
    video_list: Optional[List[Dict]] = None,
    max_comments_per_video: int = 100,
    output_csv: str | Path = "data/raw/harvested_youtube_comments.csv",
    output_json: str | Path = "data/raw/harvest_metadata.json",
) -> pd.DataFrame:
    """Execute complete harvesting workflow across verified video list."""
    api_key = load_api_key()
    videos = video_list or SEED_CLIMATE_VIDEOS

    print(f"Starting genuine YouTube comment harvest across {len(videos)} climate videos...")
    all_comments = []

    for idx, vinfo in enumerate(videos, 1):
        vid = vinfo["video_id"]
        lang = vinfo["lang"]
        ch = vinfo["channel"]
        print(f"[{idx}/{len(videos)}] Fetching comments for {ch} (video ID: {vid}, lang: {lang})...")
        c_list = fetch_comments_for_video(
            video_id=vid,
            api_key=api_key,
            max_comments=max_comments_per_video,
            expected_lang=lang,
        )
        print(f"      Retrieved {len(c_list)} valid comments.")
        all_comments.extend(c_list)

    df = pd.DataFrame(all_comments)
    if df.empty:
        print("[WARNING] No comments harvested. Check video IDs and API permissions.")
        return df

    # Deduplicate exact comment IDs
    initial_len = len(df)
    df.drop_duplicates(subset=["comment_id"], keep="first", inplace=True)
    df.drop_duplicates(subset=["video_id", "clean_text"], keep="first", inplace=True)

    dest_csv = Path(output_csv)
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest_csv, index=False, encoding="utf-8")

    # Save summary metadata
    meta = {
        "harvest_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_harvested_comments": int(initial_len),
        "unique_comments_after_dedup": int(len(df)),
        "video_count": int(df["video_id"].nunique()),
        "videos": list(df["video_id"].unique()),
        "language_distribution": df["detected_language"].value_counts().to_dict(),
        "video_distribution": df["video_id"].value_counts().to_dict(),
    }

    dest_json = Path(output_json)
    with open(dest_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\nHarvest Complete!")
    print(f"Total Unique Comments Saved : {len(df)} -> {dest_csv.resolve()}")
    print(f"Language Distribution       : {meta['language_distribution']}")
    print(f"Video Distribution          : {meta['video_distribution']}")
    return df


if __name__ == "__main__":
    harvest_comments()
