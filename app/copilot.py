"""Support Copilot engine v2: needs-human → scope (DECLINE) → known topic → route → RAG reply → checks."""
import re
from pathlib import Path

import chromadb
import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# ---- Settings ----
MODEL = "qwen2.5:7b"      # best on eval + held-out; switch to "llama3.2:3b" for a lighter, faster run
EMBED_MODEL = "nomic-embed-text"
THRESHOLD = 0.52          # max distance to the closest known question (v1: 0.39)
USE_SCOPE_CHECK = True
LENGTH_RETRY = True
MAX_WORDS = 120

DECLINE_MESSAGE = ("Sorry, I can only help with questions about your orders, payments, deliveries "
                   "and account with us. For anything else, please contact the relevant company "
                   "or service directly.")

DEFINITIONS = {
    "ACCOUNT": "creating, deleting, editing or switching an account; recovering a password; registration problems",
    "CANCEL": "asking about the FEE or penalty for cancelling (NOT actually cancelling an order)",
    "CONTACT": "wants to talk to customer service or a human agent, or asks how to contact the company",
    "DELIVERY": "what delivery options exist and how long delivery takes",
    "FEEDBACK": "making a complaint or leaving a review about the company or service",
    "INVOICE": "finding, checking or downloading an invoice or bill",
    "ORDER": "placing, changing, cancelling or tracking an order or purchase",
    "PAYMENT": "which payment methods are accepted, or problems making a payment",
    "REFUND": "refund policy, asking for money back, or tracking a refund",
    "SHIPPING": "setting up or changing the shipping address",
    "SUBSCRIPTION": "subscribing to or unsubscribing from the newsletter",
}
CATEGORIES = sorted(DEFINITIONS)
EXAMPLES = pd.read_csv(ROOT / "data" / "fewshot_examples.csv")

_collection = chromadb.PersistentClient(path=str(ROOT / "chroma_db")).get_collection("support_kb")

# Placeholders that legitimately exist in our knowledge base (e.g. {{Order Number}})
KNOWN_PLACEHOLDERS = set()
for _doc in _collection.get()["documents"]:
    KNOWN_PLACEHOLDERS |= set(re.findall(r"\{\{.*?\}\}", _doc))

# Always escalate these (whole-word matching, so "issue" does not match "sue")
RED_FLAG_PATTERNS = [r"\bfraud", r"\bscam", r"\blawyer", r"\blegal action", r"\bsue\b", r"\bcourt\b",
                     r"\bpolice\b", r"\binjur", r"\bhurt\b", r"\baccident", r"\bunauthori",
                     r"\bstolen\b", r"\bchargeback"]


def _chat(prompt, temperature=0.0):
    reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": temperature, "num_ctx": 4096})
    return reply["message"]["content"].strip()


def first_word(text):
    """Strict parsing: read only the first word of a model answer."""
    m = re.match(r"[A-Za-z]+", text.strip())
    return m.group(0).upper() if m else ""


def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def route(message):
    definitions = "\n".join(f"- {c}: {d}" for c, d in DEFINITIONS.items())
    shots = "".join(f"Message: {r.instruction}\nCategory: {r.category}\n\n"
                    for r in EXAMPLES.itertuples())
    prompt = f"""You are a customer support ticket router.
Classify the customer message into exactly ONE category.

Category definitions:
{definitions}

Examples of correctly classified messages:
{shots}
Reply with ONLY the category name. No explanation.

Customer message: {message}
Category:"""
    answer = _chat(prompt).upper()
    matches = [c for c in CATEGORIES if c in answer]
    return max(matches, key=len) if matches else "UNKNOWN"


def retrieve(message, category=None, k=3):
    where = {"category": category} if category else None
    res = _collection.query(query_embeddings=[embed(message)], n_results=k, where=where)
    return [{"question": m["question"], "intent": m["intent"], "answer": d, "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]


def needs_human(message):
    text = message.lower()
    if any(re.search(p, text) for p in RED_FLAG_PATTERNS):
        return True
    prompt = f"""Does this customer message need a HUMAN agent instead of an automated reply?
Answer YES only if the customer is very angry or abusive, threatens to complain publicly or leave,
or describes two or more different problems at once.
A single rude word or a normal request for help is NOT enough: answer NO.

Examples:
Message: my parcel is late, my refund is missing and my account is locked → YES
Message: you people are a joke, i'm telling all my friends never to buy here → YES
Message: this is unacceptable, i'll write a review everywhere about this service → YES
Message: where's my stupid receipt → NO
Message: i have a problem with my order → NO
Message: a bit annoyed my parcel is late, when will it arrive? → NO

Reply with ONLY one word: YES or NO.

Message: {message}
Answer:"""
    return first_word(_chat(prompt)) == "YES"


def in_scope(message):
    topics = "\n".join(f"- {d}" for d in DEFINITIONS.values())
    prompt = f"""You screen messages sent to the customer support of OUR online store.
Our support team handles these topics:
{topics}

Answer IN if the message is about any of these topics with our store, even with typos, rude words
or unusual wording (e.g. "penalty" or "termination fee" means a cancellation fee).
Answer OUT only if it clearly names another company, brand, app or website, or is clearly unrelated
to shopping with us (weather, sports, general knowledge, poems).
If you are not sure, answer IN.

Examples:
Message: where can i download my receipt → IN
Message: i forgot my password → IN
Message: can i speak to someone from your support team → IN
Message: my deliveroo food arrived cold → OUT
Message: where is my spotify invoice → OUT
Message: what's the score of the barcelona game → OUT

Reply with ONLY one word: IN or OUT.

Message: {message}
Answer:"""
    return first_word(_chat(prompt)) != "OUT"


def generate_reply(message, hits):
    context = "\n\n".join(f"Past customer question: {h['question']}\nApproved answer: {h['answer']}"
                          for h in hits)
    prompt = f"""You are a friendly, professional customer support agent.
Write a reply to the customer using ONLY the approved answers below as your source of information.

Rules:
- Maximum 80 words. Summarise in your own words; do not copy the approved answers.
- Do not invent policies, prices, phone numbers, emails or links.
- Keep placeholders like {{{{Order Number}}}} exactly as they are.
- If the approved answers do not cover the question, say you will pass it to a human agent.

Approved answers:
{context}

Customer message: {message}
Reply:"""
    return _chat(prompt, temperature=0.2)


def shorten(reply):
    prompt = f"""Rewrite this customer support reply in under 80 words.
Keep the meaning. Do not add any new information.
Keep placeholders like {{{{Order Number}}}} exactly as they are.

Reply:
{reply}

Shorter reply:"""
    return _chat(prompt)


def check_reply(reply, sources=None):
    problems = []
    invented = set(re.findall(r"\{\{.*?\}\}", reply)) - KNOWN_PLACEHOLDERS
    if invented:
        problems.append(f"invented placeholders: {invented}")
    if len(reply.split()) > MAX_WORDS:
        problems.append(f"too long: {len(reply.split())} words")
    return problems


def handle(message):
    """Pipeline: needs human? → in scope? → known topic? → route → retrieve → reply → checks."""
    if needs_human(message):
        return {"action": "ESCALATE", "reason": "sensitive or complex: needs a human",
                "category": route(message), "reply": None, "sources": []}

    if USE_SCOPE_CHECK and not in_scope(message):
        return {"action": "DECLINE", "reason": "off-topic or about another company",
                "category": None, "reply": DECLINE_MESSAGE, "sources": []}

    if retrieve(message)[0]["distance"] > THRESHOLD:
        return {"action": "ESCALATE", "reason": "in scope but not covered by the knowledge base",
                "category": None, "reply": None, "sources": []}

    category = route(message)
    hits = retrieve(message, category=category)
    reply = generate_reply(message, hits)
    if LENGTH_RETRY and len(reply.split()) > MAX_WORDS:
        reply = shorten(reply)

    problems = check_reply(reply, hits)
    if problems:
        return {"action": "ESCALATE", "reason": "; ".join(problems),
                "category": category, "reply": reply, "sources": hits}
    return {"action": "REPLY", "reason": "passed all checks",
            "category": category, "reply": reply, "sources": hits}
