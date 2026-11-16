import streamlit as st
from src.predict import load_artifacts, predict_message

st.set_page_config(page_title="SpamShield AI", layout="centered")

st.markdown("""
<style>
.block-container {max-width: 850px; padding-top: 2rem;}
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.25);
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

st.title("SpamShield AI")
st.caption("SMS Spam Detection Assistant")
st.divider()

try:
    _, _, _, metrics = load_artifacts()
    backend_ready = True
    backend_error = None

except FileNotFoundError as exc:
    backend_ready = False
    backend_error = str(exc)
    metrics = {}

if not backend_ready:
    st.error(f"Backend connection failed: {backend_error}")

c1, c2, c3 = st.columns(3)
with c1:
    accuracy = metrics.get("accuracy")
    st.metric(
    "Test accuracy",
    f"{accuracy * 100:.2f}%" if accuracy is not None else "Unavailable"
)
with c2:
    st.metric("Prediction confidence", "Shown per message")
with c3:
    st.metric("Backend", "Connected" if backend_ready else "Not connected")

if not backend_ready:
    st.info("Frontend is ready. Train the model to enable real predictions.")

st.divider()
st.subheader("Message checker")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Send an SMS message and I will classify it as Ham or Spam."
    }]

for item in st.session_state.messages:
    with st.chat_message(item["role"]):
        st.write(item["content"])
        result = item.get("result")
        if result:
            if result["label"] == "SPAM":
                st.error("Classification: SPAM")
            else:
                st.success("Classification: HAM")

            a, b, c = st.columns(3)
            a.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
            b.metric("Spam probability", f"{result['spam_probability'] * 100:.2f}%")
            c.metric("Ham probability", f"{result['ham_probability'] * 100:.2f}%")

prompt = st.chat_input("Enter an SMS message...")

if prompt and prompt.strip():
    st.session_state.messages.append({"role": "user", "content": prompt})

    if backend_ready:
        try:
            result = predict_message(prompt)
            reply = {
                "role": "assistant",
                "content": "Here is the analysis of your message.",
                "result": result
            }
        except Exception as exc:
            reply = {"role": "assistant", "content": f"Prediction failed: {exc}"}
    else:
        reply = {
            "role": "assistant",
            "content": "Message received, but the trained prediction backend is not connected yet."
        }

    st.session_state.messages.append(reply)
    st.rerun()

st.divider()
st.caption("SpamShield AI | NumPy Logistic Regression")
