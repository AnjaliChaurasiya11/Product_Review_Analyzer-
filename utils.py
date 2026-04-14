import re
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# Aspect synonym mapping — each keyword maps to a canonical aspect name.
# Using word-boundary regex ensures "fast" doesn't match inside "breakfast".
# ─────────────────────────────────────────────────────────────────────────────

ASPECT_SYNONYMS = {
    "Battery": [
        "battery", "batteries", "backup", "charge", "charging",
        "charge time", "mah", "power", "drain", "drains", "draining",
        "standby", "endurance", "recharge", "plug in",
    ],
    "Camera": [
        "camera", "cameras", "photo", "photos", "photograph",
        "lens", "lenses", "picture", "pictures", "shot", "shots",
        "megapixel", "portrait", "portraits", "zoom", "low light",
        "nightography", "video", "videography", "stabilization",
    ],
    "Performance": [
        "performance", "speed", "fast", "slow", "lag", "lagging",
        "sluggish", "snappy", "responsive", "processor", "chip",
        "ram", "memory", "multitasking", "smooth", "smoothly",
        "stutter", "freeze", "freezes", "throttle", "throttling",
        "benchmark", "fps", "frame rate", "framerate", "renders",
    ],
    "Price": [
        "price", "pricing", "cost", "costs", "expensive", "cheap",
        "affordable", "overpriced", "value", "worth", "budget",
        "money", "discount", "deal", "sale", "fee", "investment",
        "pay", "paying", "paid", "rupee", "dollar",
    ],
    "Display": [
        "display", "screen", "resolution", "pixel", "brightness",
        "contrast", "color", "colors", "colour", "colours", "panel",
        "oled", "amoled", "lcd", "ips", "refresh rate", "hz",
        "hdr", "nits", "glare", "uniform", "backlight",
    ],
    "Build Quality": [
        "build", "build quality", "design", "material", "materials",
        "finish", "premium", "plastic", "metal", "aluminum", "titanium",
        "sturdy", "durable", "durability", "fragile", "scratch",
        "scratches", "weight", "lightweight", "heavy", "bulky", "slim",
    ],
    "Software": [
        "software", "os", "operating system", "ui", "ux", "interface",
        "update", "updates", "bug", "bugs", "glitch", "bloatware",
        "app", "apps", "feature", "features", "settings", "usability",
        "navigation", "gesture", "gestures",
    ],
}

# ── Visual metadata for each aspect ─────────────────────────────────────────
ASPECT_META = {
    "Battery":       {"emoji": "🔋", "color": "#10b981", "bar": "green"},
    "Camera":        {"emoji": "📷", "color": "#818cf8", "bar": "purple"},
    "Performance":   {"emoji": "⚡", "color": "#f59e0b", "bar": "amber"},
    "Price":         {"emoji": "💰", "color": "#06b6d4", "bar": "cyan"},
    "Display":       {"emoji": "🖥️",  "color": "#ec4899", "bar": "pink"},
    "Build Quality": {"emoji": "🔩", "color": "#a78bfa", "bar": "violet"},
    "Software":      {"emoji": "💾", "color": "#34d399", "bar": "teal"},
}

# Legacy alias kept for backward compatibility with app.py imports
ASPECT_KEYWORDS = {aspect: synonyms for aspect, synonyms in ASPECT_SYNONYMS.items()}

# Pre-compile regex patterns for fast matching (word-boundary aware)
_ASPECT_PATTERNS: dict[str, re.Pattern] = {}
for _aspect, _synonyms in ASPECT_SYNONYMS.items():
    # Sort by length (longest first) to match multi-word phrases before single words
    sorted_syns = sorted(_synonyms, key=len, reverse=True)
    pattern = r"\b(?:" + "|".join(re.escape(s) for s in sorted_syns) + r")\b"
    _ASPECT_PATTERNS[_aspect] = re.compile(pattern, re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """Split a review paragraph into discrete sentences."""
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Aspect extraction (word-boundary regex)
# ─────────────────────────────────────────────────────────────────────────────

def extract_aspects(sentence: str) -> list[str]:
    """Return canonical aspect names mentioned in the sentence."""
    detected = []
    for aspect, pattern in _ASPECT_PATTERNS.items():
        if pattern.search(sentence):
            detected.append(aspect)
    return detected


# ─────────────────────────────────────────────────────────────────────────────
# Intelligent insight helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_insights(aspect_sentiments: dict) -> dict:
    """
    Given a dict of {aspect: {Positive, Negative, Neutral, total}},
    compute higher-level analytics:
      - most_mentioned   : aspect with highest total mentions
      - most_controversial: aspect closest to 50/50 pos/neg split
      - sentiment_narrative: natural-language summary string
      - positivity_rank  : list of aspects sorted by positive %
    """
    if not aspect_sentiments:
        return {}

    most_mentioned = max(aspect_sentiments, key=lambda a: aspect_sentiments[a]["total"])

    # Controversy score = 1 - |pos_pct - neg_pct| / 100  (1 = perfectly split)
    def controversy(asp):
        total = aspect_sentiments[asp]["total"] or 1
        pos_pct = aspect_sentiments[asp]["Positive"] / total
        neg_pct = aspect_sentiments[asp]["Negative"] / total
        return 1 - abs(pos_pct - neg_pct)

    most_controversial = max(aspect_sentiments, key=controversy)

    # Positivity ranking
    def pos_ratio(asp):
        total = aspect_sentiments[asp]["total"] or 1
        return aspect_sentiments[asp]["Positive"] / total

    positivity_rank = sorted(aspect_sentiments.keys(), key=pos_ratio, reverse=True)

    # Build a plain-English narrative
    loved = [a for a in positivity_rank if pos_ratio(a) >= 0.6]
    disliked = [a for a in positivity_rank if pos_ratio(a) < 0.4]
    narrative_parts = []
    if loved:
        narrative_parts.append(f"Users love **{' & '.join(loved)}**")
    if disliked:
        narrative_parts.append(f"but complain about **{' & '.join(disliked)}**")
    if not narrative_parts:
        narrative_parts.append("Sentiment is mixed across all aspects")
    sentiment_narrative = " ".join(narrative_parts) + "."

    return {
        "most_mentioned": most_mentioned,
        "most_controversial": most_controversial,
        "controversy_score": round(controversy(most_controversial) * 100),
        "positivity_rank": positivity_rank,
        "sentiment_narrative": sentiment_narrative,
        "loved_aspects": loved,
        "disliked_aspects": disliked,
    }
