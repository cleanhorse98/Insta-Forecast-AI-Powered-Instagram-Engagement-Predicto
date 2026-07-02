import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from dateutil import parser
from datetime import timedelta
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --------------------------
# SMART DATETIME PARSER
# --------------------------
ISO_YEAR_FIRST_RE = re.compile(r"^\s*\d{4}[-/T]\d{1,2}[-/T]\d{1,2}")
DMY_RE = re.compile(r"^\s*\d{1,2}[-/\/]\d{1,2}[-/\/]\d{2,4}")
EMPTY_RE = re.compile(r"^\s*$")

def safe_parse_dt(value):
    if pd.isna(value):
        return pd.NaT
    s = str(value).strip()
    if EMPTY_RE.match(s):
        return pd.NaT
    try:
        if ISO_YEAR_FIRST_RE.match(s):
            return parser.parse(s, yearfirst=True)
        if DMY_RE.match(s):
            return parser.parse(s, dayfirst=True)
        return parser.parse(s, dayfirst=True)
    except Exception:
        try:
            return parser.parse(s)
        except Exception:
            return pd.NaT

def normalize_series_to_utc_naive(series):
    series = pd.to_datetime(series, errors="coerce")
    try:
        if series.dt.tz is not None:
            series = series.dt.tz_convert("UTC").dt.tz_localize(None)
    except Exception:
        def _utc(x):
            try:
                if getattr(x, "tzinfo", None) is not None:
                    return x.astimezone(tz=parser.tz.UTC).replace(tzinfo=None)
            except:
                pass
            return x
        series = series.apply(_utc)
        series = pd.to_datetime(series, errors="coerce")
    return series

# --------------------------
# TEXT CLEANING
# --------------------------
def clean_caption(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"#\w+", " hashtag_token ", text)
    text = re.sub(r"@\w+", " mention_token ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()

def extract_hashtags(hs):
    if not isinstance(hs, str):
        return []
    parts = re.split(r"[,\s]+", hs.strip())
    return [p.lower() for p in parts if p.strip()]

POS = {"good","great","best","love","awesome","happy","win","success"}
NEG = {"bad","sad","terrible","hate","worst","lose","fail"}

def simple_sentiment(text):
    words = re.findall(r"\w+", text.lower())
    if len(words) == 0:
        return 0.0
    pos = sum(w in POS for w in words)
    neg = sum(w in NEG for w in words)
    return (pos - neg) / len(words)

# --------------------------
# LOAD INPUT
# --------------------------
INPUT_FILE = "output_fixed.xlsx"
OUTPUT_FILE = "processed1.csv"

df = pd.read_excel(INPUT_FILE)
print("Loaded rows:", len(df))

# --------------------------
# NLP PROCESSING
# --------------------------
df["caption_clean"] = df.get("caption", "").fillna("").apply(clean_caption)
df["hashtags_list"] = df.get("hashtags", "").fillna("").apply(extract_hashtags)
df["hashtags_clean"] = df["hashtags_list"].apply(lambda x: " ".join(x))

df["sentiment_score"] = df["caption_clean"].apply(simple_sentiment)

df["combined_text"] = (df["caption_clean"] + " " + df["hashtags_clean"]).str.strip()
texts = df["combined_text"].tolist()

# --------------------------
# EMBEDDINGS (TF-IDF + SVD)
# --------------------------
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X = tfidf.fit_transform(texts)

svd = TruncatedSVD(n_components=256, random_state=42)
embed = svd.fit_transform(X)

embed_df = pd.DataFrame(embed, columns=[f"embed_{i}" for i in range(256)])
df = pd.concat([df.reset_index(drop=True), embed_df.reset_index(drop=True)], axis=1)

# --------------------------
# FIX is_video TO BINARY
# --------------------------
def to_binary_is_video(v):
    if pd.isna(v):
        return 0
    if isinstance(v, (int, np.integer)):
        return int(bool(v))
    s = str(v).strip().lower()
    if s in ("1","true","t","yes","y"):
        return 1
    return 0

df["is_video"] = df["is_video"].apply(to_binary_is_video)

# --------------------------
# FIX video_view_count
# --------------------------
df["video_view_count"] = pd.to_numeric(df.get("video_view_count", np.nan), errors="coerce")

# missing video views:
# → if video → 1
# → if not video → 0
df.loc[df["video_view_count"].isna() & (df["is_video"] == 1), "video_view_count"] = 1
df.loc[df["video_view_count"].isna() & (df["is_video"] == 0), "video_view_count"] = 0

df["video_view_count"] = df["video_view_count"].astype(int)

# --------------------------
# TIME FEATURES (TRUE LOGIC)
# --------------------------
df["timestamp_parsed"] = df["timestamp_utc"].apply(safe_parse_dt)
df["scraped_parsed"] = df["scraped_at_utc"].apply(safe_parse_dt)

df["timestamp_norm"] = normalize_series_to_utc_naive(df["timestamp_parsed"])
df["scraped_norm"] = normalize_series_to_utc_naive(df["scraped_parsed"])

df["scraped_norm"] = df["scraped_norm"].dt.floor("min")

mask = (df["timestamp_norm"].notna()) & (df["scraped_norm"].notna()) & (df["timestamp_norm"] > df["scraped_norm"])
if mask.any():
    logging.info(f"Adjusting scraped_at for {mask.sum()} rows where timestamp > scraped_at")
    df.loc[mask, "scraped_norm"] = df.loc[mask, "timestamp_norm"] + timedelta(days=1)

df["post_age_hours"] = (
    (df["scraped_norm"] - df["timestamp_norm"])
    .dt.total_seconds()
    .div(3600)
    .clip(lower=0)
).fillna(0)

# --------------------------
# FINAL OUTPUT COLUMNS
# --------------------------
final_cols = [
    "shortcode",
    "is_video",
    "likes",
    "comments_count",
    "video_view_count",
    "seo_caption_len",
    "seo_word_count",
    "seo_hashtag_count",
    "seo_mention_count",
    "seo_recency_norm",
    "seo_engagement_norm",
    "seo_hashtag_norm",
    "seo_seo_score",
    "post_age_hours",
    "sentiment_score",
] + [f"embed_{i}" for i in range(256)]

df_final = df[final_cols].copy()

df_final.to_csv(OUTPUT_FILE, index=False)
print("✔ DONE! Saved", OUTPUT_FILE)
