import streamlit as st
from core.parser import load_transactions
from core.embeddings import load_model, get_embeddings
from core.vector_store import VectorStore
from core.gemini import get_answer

st.set_page_config(page_title="Finance Assistant", page_icon="💰")
st.title("💰 Personal Finance Assistant")
st.caption("Upload your transactions and ask questions about your spending")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

uploaded_file = st.file_uploader("Upload your transactions CSV", type=["csv"])

if uploaded_file:
    if st.session_state.vector_store is None:
        with st.spinner("Processing your transactions..."):
            try:
                chunks = load_transactions(uploaded_file)
                if len(chunks) == 0:
                    st.error("No transactions found in CSV. Please check your file format.")
                else:
                    embeddings = get_embeddings(chunks)
                    vs = VectorStore()
                    vs.build(chunks, embeddings)
                    st.session_state.vector_store = vs
                    st.success(f"✅ Loaded {len(chunks)} transactions successfully!")
            except Exception as e:
                st.error(f"❌ Error reading CSV: make sure it has date, amount, category, description columns.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("Ask about your spending..."):
    if st.session_state.vector_store is None:
        st.warning("Please upload a CSV file first!")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                query_embedding = get_embeddings([question])[0]
                relevant_chunks = st.session_state.vector_store.search(query_embedding)
                answer = get_answer(question, relevant_chunks)
                st.write(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})