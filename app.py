import streamlit as st
from src.inference.semantic_engine import SemanticEngine

engine = SemanticEngine()


# =====================
# HELPERS
# =====================
def format_green(tokens, energy, co2):
    energy_mwh = energy * 1e6
    co2_mg = co2 * 1e6

    return (
        f"{tokens} tokens saved, "
        f"{energy_mwh:.4f} mWh, "
        f"{co2_mg:.2f} mg CO₂"
    )


def energy_to_phone_seconds(energy_kwh):
    phone_power_w = 10
    energy_wh = energy_kwh * 1000
    seconds = (energy_wh / phone_power_w) * 3600
    return seconds


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
    "selected_cached_question": None,

    # dashboard
    "total_tokens_saved": 0,
    "total_energy_saved": 0.0,
    "total_co2_saved": 0.0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =====================
# HEADER + DASHBOARD
# =====================
header = st.container()

with header:
    st.markdown("# 🍃 Green AI App")

    st.markdown(
        """
        <div style="
            background-color: rgba(255,255,255,0.03);
            padding: 14px 18px;
            border-radius: 12px;
            margin-bottom: 10px;
            border: 1px solid rgba(255,255,255,0.08);
        ">
        <b>🌱 Session Impact</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    d1, d2, d3, d4 = st.columns(4)

    d1.metric("Tokens", st.session_state.total_tokens_saved)

    energy_mwh_total = st.session_state.total_energy_saved * 1e6
    d2.metric("Energy (mWh)", f"{energy_mwh_total:.4f}")

    co2_mg_total = st.session_state.total_co2_saved * 1e6
    d3.metric("CO₂ (mg)", f"{co2_mg_total:.2f}")

    phone_seconds_total = energy_to_phone_seconds(
        st.session_state.total_energy_saved
    )
    d4.metric("Phone 🔋 (sec)", f"{phone_seconds_total:.2f}")

    st.markdown("---")


# =====================
# CHAT DISPLAY
# =====================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# =====================
# LLM WAIT
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

                # update dashboard
                st.session_state.total_tokens_saved += item["tokens_saved"]
                st.session_state.total_energy_saved += item["energy_saved"]
                st.session_state.total_co2_saved += item["co2_saved"]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"{item['answer']}<br><small><b>Retrieved from cache ("
                        f"{format_green(item['tokens_saved'], item['energy_saved'], item['co2_saved'])}"
                        f")</b></small>"
                    )
                })

                st.session_state.last_answer = item["answer"]
                st.session_state.last_source = "cache"
                st.session_state.selected_cached_question = item["question"]
                st.session_state.mode = "feedback"

                st.rerun()

    if data["drifted"]:
        st.markdown("<small><b>⚠️ Possible (drift detected)</b></small>", unsafe_allow_html=True)
        for item in data["drifted"]:
            if st.button(item["question"]):

                st.session_state.total_tokens_saved += item["tokens_saved"]
                st.session_state.total_energy_saved += item["energy_saved"]
                st.session_state.total_co2_saved += item["co2_saved"]

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": (
                        f"{item['answer']}<br><small><b>Retrieved from cache ("
                        f"{format_green(item['tokens_saved'], item['energy_saved'], item['co2_saved'])}"
                        f")</b></small>"
                    )
                })

                st.session_state.last_answer = item["answer"]
                st.session_state.last_source = "cache"
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

    cached_q = st.session_state.selected_cached_question
    original_q = st.session_state.question

    cached_tokens = engine.count_tokens(cached_q)
    original_tokens = engine.count_tokens(original_q)

    delta_tokens = max(original_tokens - cached_tokens, 0)

    energy_saved = delta_tokens * engine.ENERGY_PER_TOKEN
    co2_saved = energy_saved * engine.CARBON_INTENSITY

    if col1.button(f"Reuse prompt:\n{cached_q}"):

        # add reuse savings
        st.session_state.total_tokens_saved += delta_tokens
        st.session_state.total_energy_saved += energy_saved
        st.session_state.total_co2_saved += co2_saved

        st.session_state.conversation = [
            {"role": "user", "content": cached_q}
        ]
        st.session_state.mode = "llm"
        st.rerun()

    if col2.button(f"Use original:\n{original_q}"):
        st.session_state.conversation = [
            {"role": "user", "content": original_q}
        ]
        st.session_state.mode = "llm"
        st.rerun()

    st.markdown(
        f"<small><b>With reuse prompt you can save "
        f"{delta_tokens} tokens, "
        f"{energy_saved * 1e6:.4f} mWh, "
        f"{co2_saved * 1e6:.2f} mg CO₂"
        f"</b></small>",
        unsafe_allow_html=True
    )


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

        # update dashboard
        st.session_state.total_tokens_saved += result["tokens_saved"]
        st.session_state.total_energy_saved += result["energy_saved"]
        st.session_state.total_co2_saved += result["co2_saved"]

        st.session_state.messages.append({
            "role": "assistant",
            "content": (
                f"{result['answer']}<br><small><b>Retrieved from cache ("
                f"{format_green(result['tokens_saved'], result['energy_saved'], result['co2_saved'])}"
                f")</b></small>"
            )
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