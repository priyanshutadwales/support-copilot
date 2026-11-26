## v2: evaluating the whole copilot (not just routing)
A 40-message eval set with the expected behaviour written in advance, in five types: in-scope tickets, messy (typos), off-topic, other companies, and sensitive (angry, legal, fraud). Three actions: **REPLY**, **DECLINE** (polite fixed message, no human needed) and **ESCALATE** (human).

| Version | What changed | Eval set | Notes |
|---|---|---|---|
| v2_0 | v1 pipeline | 62% | Typos treated as off-topic; long replies blocked; other companies answered |
| v2_1–2_3 | Relaxed threshold, length retry, scope check | 75% → 50% | Each fix worked alone but removed an accidental safety net: sensitive dropped to 40% |
| v2_4–2_5 | Few-shot checks, keyword safety net, whole-word matching | 82% → 88% | Found substring bugs ("sue" inside "issue"); 3B model kept trading one error for another |
| v2_6 | Same prompts, qwen2.5:7b | 95% | Bigger model fixed what prompt tweaks couldn't |
| v2_7 | Placeholder whitelist from the knowledge base | **98%** | Only {{...}} variables that exist nowhere in the KB count as invented |

**Held-out test (15 new messages, never used for tuning):** qwen2.5:7b **87%**, llama3.2:3b **93%** (a one-message difference, so within noise). **100% on sensitive, off-topic and other-company messages**, with zero dangerous errors: every failure was either too cautious or caused by garbled data. About 3 s per message on a laptop.

**Lessons:**
- Re-run the whole eval after every change: fixes can quietly break other behaviour.
- Not all errors cost the same: answering a legal threat is worse than escalating a normal question.
- Parse LLM output strictly and match keywords as whole words.
- Know when to stop prompt-tweaking: if fixes keep moving the problem, try a better model.
- Tuning on the same test inflates the score (98% vs 87–93% held-out). Report the held-out number.

# 🎧 Support Copilot

An AI customer-support copilot that **routes** tickets, **answers** them from approved replies (RAG), and **escalates** to a human when it shouldn't answer. Runs fully locally with Llama 3.2 3B via Ollama.

## The problem
Support teams receive thousands of messages a day. Each one must reach the right team, simple questions should be answered instantly, and sensitive or unclear ones must go to a human. Wrong automatic answers cost customer trust.

## How it works
```
Customer message
  → Guardrail: needs a human? (angry, legal, several problems) → ESCALATE
  → Guardrail: anything similar in the knowledge base?          → no → ESCALATE
  → Route to 1 of 11 categories (LLM, definitions + few-shot)
  → Retrieve 3 similar past tickets from that category (Chroma)
  → Generate a reply using ONLY approved answers
  → Output checks: invented placeholders? too long?             → ESCALATE with draft
  → REPLY
```

## Results: routing (99-ticket stratified test set, same tickets for every version)
| Approach | Labelled examples used | Test set 1 | Fresh test set 2 | Speed |
|---|---|---|---|---|
| LLM zero-shot (category names only) | 0 | 38.4% | — | ~170 ms |
| + one-line category definitions | 0 | 78.8% | — | ~180 ms |
| + 22 few-shot examples | 22 | 98.0% | 93.9% | ~160 ms |
| scikit-learn classifier (TF-IDF + logistic regression) | 26,610 | 98.0% | 99.0% | <1 ms |
| scikit-learn, character n-grams | 26,610 | 100% | 100% | <1 ms |

## Key findings
- **Most of the zero-shot failure was the prompt, not the model.** Labels like CANCEL (= cancellation *fee*) and SUBSCRIPTION (= *newsletter*) were ambiguous. Adding one-line definitions doubled accuracy with no training.
- **With lots of labelled data, a simple classifier wins** on speed and accuracy. The LLM reached ~94–98% with only 22 examples, which matters because most companies don't have tens of thousands of labelled tickets.
- **Classifiers match words; LLMs understand meaning.** On a hand-made challenge set, only the LLM understood "I have to give an item back" as a refund. The word classifier broke on typos in key words ("cash bac"); character n-grams fixed that.
- **The category list itself had gaps:** no category for returns or off-topic requests, so every model had to pick a wrong one.
- **Guardrails catch only what they're designed for.** Testing found an invented placeholder (`{{Weather Website}}`), now caught by a code check, and a request about another company ("place an order on Amazon") that slipped through: it looked on-topic to the similarity check, was routed to DELIVERY, and got a reply with a small hallucination. Fix planned for v2.

## Limitations (honest notes)
- The Bitext dataset is templated and cleaner than real tickets; expect lower accuracy on real customer messages.
- The 0.39 retrieval-distance threshold was tuned on a small sample.
- Complaints phrased as "a claim against your business" are sometimes declined as off-topic (seen in both test sets); planned fix in v3 with new test messages.
- Eval sets are small (40 + 15 messages); differences of one or two messages are within noise.
- Claims not supported by the sources (e.g. "the steps I provided earlier") are not caught yet. Next: an LLM-as-judge groundedness check.

## Run it locally
1. Install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b` and `ollama pull nomic-embed-text`
2. `python3.12 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
3. Run `notebooks/01_explore_data.ipynb`, then `notebooks/02_routing.ipynb` (downloads the data, runs the evals, builds the knowledge base)
4. `streamlit run app/streamlit_app.py`

## Data
[Bitext Customer Support dataset](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) (26,872 messages, 11 categories, 27 intents), licensed CDLA-Sharing-1.0.

## Next steps (v2)
- Detect requests about other companies, and add a groundedness (LLM-as-judge) check
- Compare with a larger local model and with Jev (a System One decision model)
- Hybrid routing: classifier when confident, LLM otherwise
- Deploy online and collect real user feedback
