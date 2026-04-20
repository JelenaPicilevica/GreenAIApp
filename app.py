import streamlit as st
from src.inference.semantic_engine import SemanticEngine

engine = SemanticEngine()

# =====================
# STATE INIT
# =====================
defaults = {
    "messages": [],
    "mode": "idle",
    "candidates": None,
    "conversation": [],
    "last_answer": None,
    "last_source": None,
    "question": None,
    "selected_cached_question": None
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# =====================
st.title("🍃 Green AI App")

# =====================
# CHAT DISPLAY
# =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# =====================
# LLM WAIT MESSAGE
# =====================
if st.session_state.mode == "llm_wait":
    st.markdown(
        "<small><b>Please provide more details so I can help you better.</b></small>",
        unsafe_allow_html=True
    )

# =====================
# CANDIDATES
# =====================
if st.session_state.mode == "candidates":

    data = st.session_state.candidates

    if data["no_drift"]:
        st.markdown("<small><b>Closest matches</b></small>", unsafe_allow_html=True)
        for item in data["no_drift"]:
            if st.button(item["question"]):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{item['answer']}<br><small><b>Retrieved from cache</b></small>"
                })
                st.session_state.last_answer = item["answer"]
                st.session_state.last_source = "cache"

                # STORE SELECTED QUESTION
                st.session_state.selected_cached_question = item["question"]

                st.session_state.mode = "feedback"
                st.rerun()

    if data["drifted"]:
        st.markdown("<small><b>⚠️ Possible (drift detected)</b></small>", unsafe_allow_html=True)
        for item in data["drifted"]:
            if st.button(item["question"]):
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"{item['answer']}<br><small><b>Retrieved from cache</b></small>"
                })
                st.session_state.last_answer = item["answer"]
                st.session_state.last_source = "cache"

                # STORE SELECTED QUESTION
                st.session_state.selected_cached_question = item["question"]

                st.session_state.mode = "feedback"
                st.rerun()

    if st.button("None of these"):
        st.session_state.mode = "llm"
        st.session_state.conversation = [
            {"role": "user", "content": st.session_state.question}
        ]
        st.rerun()

# =====================
# CHOOSE PROMPT
# =====================
if st.session_state.mode == "choose_prompt":

    st.markdown("Choose how to proceed:")

    col1, col2 = st.columns(2)

    if col1.button(f"Reuse prompt: {st.session_state.selected_cached_question}"):
        st.session_state.conversation = [
            {"role": "user", "content": st.session_state.selected_cached_question}
        ]
        st.session_state.mode = "llm"
        st.rerun()

    if col2.button(f"Use original: {st.session_state.question}"):
        st.session_state.conversation = [
            {"role": "user", "content": st.session_state.question}
        ]
        st.session_state.mode = "llm"
        st.rerun()

# =====================
# LLM FLOW
# =====================
if st.session_state.mode == "llm":

    answer = engine.call_llm(st.session_state.conversation)

    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{answer}<br><small><b>Retrieved via LLM</b></small>"
    })

    st.session_state.conversation.append({
        "role": "assistant",
        "content": answer
    })

    st.session_state.last_answer = answer
    st.session_state.last_source = "llm"

    if engine.is_clarification(answer):
        st.session_state.mode = "llm_wait"
    else:
        st.session_state.mode = "feedback"

    st.rerun()

# =====================
# FEEDBACK
# =====================
if st.session_state.mode == "feedback":

    st.markdown("Are you satisfied or do you want to continue with LLM?")

    c1, c2 = st.columns(2)

    if c1.button("Yes"):

        if st.session_state.last_source == "llm":
            engine.save_to_cache(
                st.session_state.conversation,
                st.session_state.last_answer
            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": "Saved to cache ✅"
            })

        st.session_state.mode = "idle"
        st.session_state.conversation = []
        st.rerun()

    if c2.button("Continue"):

        if st.session_state.last_source == "cache":
            st.session_state.mode = "choose_prompt"
        else:
            st.session_state.mode = "llm_wait"

        st.rerun()

# =====================
# INPUT
# =====================
user_input = st.chat_input("Ask something...")

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    if st.session_state.mode in ["llm", "llm_wait"]:
        st.session_state.conversation.append({
            "role": "user",
            "content": user_input
        })
        st.session_state.mode = "llm"
        st.rerun()

    st.session_state.question = user_input

    result = engine.analyze(user_input)

    if result["status"] == "auto":
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{result['answer']}<br><small><b>Retrieved from cache</b></small>"
        })
        st.session_state.last_answer = result["answer"]
        st.session_state.last_source = "cache"
        st.session_state.selected_cached_question = user_input
        st.session_state.mode = "feedback"
        st.rerun()

    if result["status"] == "candidates":
        st.session_state.candidates = result
        st.session_state.mode = "candidates"
        st.rerun()

    st.session_state.conversation = [
        {"role": "user", "content": user_input}
    ]
    st.session_state.mode = "llm"
    st.rerun()