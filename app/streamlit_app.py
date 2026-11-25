import time

import streamlit as st

from copilot import handle

st.set_page_config(page_title="Support Copilot", page_icon="🎧")
st.title("🎧 Support Copilot")
st.caption("Routes customer messages, answers from approved replies, and escalates to a human "
           "when unsure. Runs locally with Llama 3.2 3B via Ollama.")

message = st.text_area("Customer message",
                       placeholder="e.g. I was charged twice for my order and want my money back")

if st.button("Handle message", type="primary") and message.strip():
    with st.spinner("Thinking..."):
        start = time.perf_counter()
        result = handle(message)
        seconds = time.perf_counter() - start

    c1, c2, c3 = st.columns(3)
    c1.metric("Action", result["action"])
    c2.metric("Category", result["category"] or "—")
    c3.metric("Time", f"{seconds:.1f} s")

    if result["action"] == "REPLY":
        st.success("✅ Automated reply")
        st.write(result["reply"])
    else:
        st.warning(f"🙋 Escalated to a human agent: {result['reason']}")
        if result["reply"]:
            st.markdown("**Draft reply for the agent to review:**")
            st.info(result["reply"])

    if result["sources"]:
        with st.expander("📚 Sources used"):
            for h in result["sources"]:
                st.markdown(f"**{h['intent']}** · distance {h['distance']}  \n*{h['question']}*")
