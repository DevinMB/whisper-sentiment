import os
import json
import time
import re
import logging
import sys
from dotenv import load_dotenv
from kafka import KafkaConsumer, KafkaProducer
import requests

load_dotenv()

# ---------------- ENV ----------------
APP_NAME               = os.getenv("APP_NAME", "whispers-sentiment-enricher")
KAFKA_BOOTSTRAP        = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_INPUT_TOPIC      = os.getenv("KAFKA_INPUT_TOPIC", "whispererer-log")
KAFKA_OUTPUT_TOPIC     = os.getenv("KAFKA_OUTPUT_TOPIC", "whispererer-log-enriched")
GROUP_ID               = os.getenv("KAFKA_GROUP_ID", "whispers-sentiment-enricher")
AUTO_OFFSET_RESET      = os.getenv("AUTO_OFFSET_RESET", "latest") 

OLLAMA_ENABLED         = os.getenv("OLLAMA_ENABLED", "true").strip().lower() == "true"
OLLAMA_BASE_URL        = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL           = os.getenv("OLLAMA_MODEL", "phi3:mini")

LOG_LEVEL              = os.getenv("LOG_LEVEL", "INFO").upper()
TARGET_CHAT_KEY        = os.getenv("TARGET_CHAT_KEY", "The Whispers 2")

logging.basicConfig(
    stream=sys.stdout,
    level=LOG_LEVEL,
    format=f"%(asctime)s | {APP_NAME} | %(levelname)s | %(message)s"
)

# ---------------- BUCKETS (precedence order matters) ----------------
BUCKETS = [
    "horny",
    "party_substance",
    "food_craving",
    "question_help",
    "banter_meme",
    "neutral_other",
]

ORDERED_REGEX = [
    ("horny", re.compile(r"\b(horny|thirsty|arous(e|ed)|sext|nsfw|turned on|moist|nut(ting)?|pussy|fuck|eat out|suck|ride|bang|69|slut|nude|booty|🍆|😈)\b", re.I)),
    ("party_substance", re.compile(r"\b(weed|pen\b|vape|joint|blunt|baja\s*blast(ed)?|drunk|shots?|tequila|stoned|high)\b", re.I)),
    ("food_craving", re.compile(r"\b(mcgriddles?|mcdonald'?s|taco\s*bell|pizza|snack|coffee|caffeine|nectar of the gods|hungry|craving|eat|drink)\b", re.I)),
    ("question_help", re.compile(r"\?\s*$|^(how|what|why|who|where|when|can|does|do|is|are)\b", re.I)),
    ("banter_meme", re.compile(r"\b(amen|lol|lmao|rofl|bit|meme|never forget|i('m| am) jesus|sacrifices were made)\b", re.I)),
]

SYSTEM_PROMPT = (
    "You are a precise JSON-only classifier for short, informal chat messages. "
    "Classify the message into exactly one of these buckets: "
    f"{', '.join(BUCKETS)}. "
    "Return strict JSON only with this shape: "
    "{\"bucket\":\"<one_of_buckets>\",\"score\":<0..1>,\"reason\":\"<short>\"}. "
    "Do not include any extra text besides the JSON."
)

# ---------------- CLASSIFICATION ----------------
def regex_classify_first(text: str):
    if not text:
        return None
    for bucket, rx in ORDERED_REGEX:
        if rx.search(text):
            return bucket
    return None

def classify_with_regex(text: str):
    bucket = regex_classify_first(text or "")
    if bucket:
        conf = 0.85 if bucket == "horny" else 0.7
        return {
            "bucket": bucket,
            "score": conf,
            "reason": f"regex:{bucket}",
            "model": "regex",
            "confidence": conf,
        }
    return {
        "bucket": "neutral_other",
        "score": 0.55,
        "reason": "regex:none",
        "model": "regex",
        "confidence": 0.55,
    }

def classify_with_ollama(text: str):
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nMessage:\n{text}\nJSON:"
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 128},
                "format": "json",   # ask for JSON-only tokens
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        raw = (data.get("response") or "").strip()

        # Robust parse: try JSON, else extract first {...}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.S)
            if not m:
                raise
            parsed = json.loads(m.group(0))

        bucket = parsed.get("bucket")
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")

        if bucket not in BUCKETS:
            raise ValueError(f"Invalid bucket from model: {bucket}")

        # Precedence override: horny beats everything if regex says so
        if regex_classify_first(text) == "horny" and bucket != "horny":
            bucket = "horny"
            score = max(score, 0.9)
            reason = (reason + " (regex horny override)").strip()

        return {"bucket": bucket, "score": score, "reason": reason,
                "model": OLLAMA_MODEL, "confidence": score}
    except Exception as e:
        logging.warning(f"Ollama classification failed ({e}); using regex fallback.")
        return classify_with_regex(text)

def classify(text: str):
    if OLLAMA_ENABLED:
        return classify_with_ollama(text)
    return classify_with_regex(text)

# ---------------- KAFKA ----------------
def build_consumer():
    # No consumer_timeout_ms => blocking poll via .poll()
    return KafkaConsumer(
        KAFKA_INPUT_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        group_id=GROUP_ID,
        auto_offset_reset=AUTO_OFFSET_RESET,
        enable_auto_commit=True,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda v: v.decode("utf-8") if v else None,
        max_poll_records=200,
        request_timeout_ms=40000,
        session_timeout_ms=15000,
        heartbeat_interval_ms=3000,
    )

def build_producer():
    # On single broker, acks='all' works if min.insync.replicas=1; otherwise use acks='1'
    return KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        acks="all",
        linger_ms=10,
        retries=10,
        max_in_flight_requests_per_connection=1,  # safer retries
        value_serializer=lambda v: json.dumps(v, separators=(",", ":")).encode("utf-8"),
        key_serializer=lambda v: v.encode("utf-8") if v else None,
        request_timeout_ms=20000,
        retry_backoff_ms=200,
    )

# ---------------- MAIN LOGIC ----------------
def enrich_record(doc: dict):
    text = doc.get("message") or doc.get("text") or doc.get("content") or ""
    sentiment = classify(text)
    enriched = dict(doc)
    enriched.update({
        "sentiment_bucket": sentiment["bucket"],
        "sentiment_score": sentiment["score"],
        "sentiment_reason": sentiment["reason"],
        "sentiment_model": sentiment["model"],
        "confidence": sentiment["confidence"],
        "classified_at": time.time(),
    })
    return enriched

def main():
    logging.info(f"Starting {APP_NAME}")
    logging.info(f"Input topic: {KAFKA_INPUT_TOPIC}  ->  Output topic: {KAFKA_OUTPUT_TOPIC}")
    logging.info(f"Filter: only process records with key == '{TARGET_CHAT_KEY}'")
    logging.info(f"Ollama enabled: {OLLAMA_ENABLED} ({OLLAMA_MODEL if OLLAMA_ENABLED else 'regex-only'})")
    logging.info("Reminder: Single-broker cluster → topic RF=1 and min.insync.replicas=1")

    consumer = build_consumer()
    producer = build_producer()

    while True:
        msg_pack = consumer.poll(timeout_ms=1000, max_records=100)
        if not msg_pack:
            continue

        for _tp, records in msg_pack.items():
            for msg in records:
                try:
                    if msg.key != TARGET_CHAT_KEY:
                        continue

                    doc = msg.value
                    if not isinstance(doc, dict):
                        logging.warning("Skipping non-JSON message")
                        continue

                    timestamp = doc.get("timestamp")
                    if timestamp is None:
                        logging.warning("Message missing timestamp, skipping")
                        continue
                    try:
                        ts_key = str(int(float(timestamp)))
                    except Exception:
                        logging.warning(f"Unusable timestamp '{timestamp}', skipping")
                        continue

                    enriched = enrich_record(doc)

                    # Send and wait for broker ack (surface real errors)
                    future = producer.send(KAFKA_OUTPUT_TOPIC, key=ts_key, value=enriched)
                    future.get(timeout=10)

                    logging.info(
                        f"Processed [{TARGET_CHAT_KEY}] ts={ts_key} -> bucket={enriched['sentiment_bucket']}"
                    )
                except Exception as e:
                    logging.exception(f"Error processing message: {e}")

if __name__ == "__main__":
    main()

