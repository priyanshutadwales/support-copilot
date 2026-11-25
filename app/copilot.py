"""Support Copilot engine: routing + RAG + guardrails, running locally with Ollama."""
import re
from pathlib import Path

import chromadb
import ollama
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL = "llama3.2:3b"
EMBED_MODEL = "nomic-embed-text"
THRESHOLD = 0.39   # tuned on a small sample: in-scope max 0.322, off-topic min 0.450
MAX_WORDS = 120

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


def _chat(prompt, temperature=0.0):
    reply = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}],
                        options={"temperature": temperature})
    return reply["message"]["content"].strip()


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
    prompt = f"""Does this customer message need a HUMAN agent instead of an automated reply?
Answer YES if ANY of these is true:
- the customer is very angry, insulting, or threatening to leave or complain publicly
- it mentions legal action, a lawyer, fraud, a safety issue, or an injury
- it describes several different problems at once

Otherwise answer NO.
Reply with ONLY one word: YES or NO.

Customer message: {message}
Answer:"""
    return "YES" in _chat(prompt).upper()


def generate_reply(message, hits):
    context = "\n\n".join(f"Past customer question: {h['question']}\nApproved answer: {h['answer']}"
                          for h in hits)
    prompt = f"""You are a friendly, professional customer support agent.
Write a reply to the customer using ONLY the approved answers below as your source of information.

Rules:
- Keep it short: maximum {MAX_WORDS} words.
- Do not invent policies, prices, phone numbers, emails or links.
- Keep placeholders like {{{{Order Number}}}} exactly as they are.
- If the approved answers do not cover the question, say you will pass it to a human agent.

Approved answers:
{context}

Customer message: {message}
Reply:"""
    return _chat(prompt, temperature=0.2)


def check_reply(reply, sources):
    problems = []
    allowed = set()
    for h in sources:
        allowed |= set(re.findall(r"\{\{.*?\}\}", h["answer"]))
    invented = set(re.findall(r"\{\{.*?\}\}", reply)) - allowed
    if invented:
        problems.append(f"invented placeholders: {invented}")
    if len(reply.split()) > MAX_WORDS:
        problems.append(f"too long: {len(reply.split())} words")
    return problems


def handle(message):
    """Full pipeline: guardrails in → route → retrieve → generate → guardrails out."""
    if needs_human(message):
        return {"action": "ESCALATE", "reason": "sensitive or complex: needs a human",
                "category": route(message), "reply": None, "sources": []}

    if retrieve(message)[0]["distance"] > THRESHOLD:
        return {"action": "ESCALATE", "reason": "no matching knowledge (off-topic or not covered)",
                "category": None, "reply": None, "sources": []}

    category = route(message)
    hits = retrieve(message, category=category)
    reply = generate_reply(message, hits)
    problems = check_reply(reply, hits)
    if problems:
        return {"action": "ESCALATE", "reason": "; ".join(problems),
                "category": category, "reply": reply, "sources": hits}
    return {"action": "REPLY", "reason": "passed all checks",
            "category": category, "reply": reply, "sources": hits}
