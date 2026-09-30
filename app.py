import streamlit as st
import numpy as np
import re

from google import genai
from google.genai import types

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================
# PAGE SETTINGS
# =========================

st.set_page_config(
    page_title="AI Knowledge Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Knowledge Agent")
st.write(
    "Upload any TXT file and ask questions about its content "
    "or ask general questions."
)


# =========================
# GEMINI API
# =========================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )

except Exception:
    st.error(
        "❌ GEMINI_API_KEY not found in Streamlit Secrets."
    )
    st.stop()


# =========================
# FILE UPLOAD
# =========================

uploaded_file = st.file_uploader(
    "📂 Upload your TXT File",
    type=["txt"]
)


# =========================
# TEXT CLEANING
# =========================

def clean_text(text):

    text = text.lower()

    replacements = {
        "ai": "artificial intelligence",
        "ml": "machine learning",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "llm": "large language model",
        "cv": "computer vision",
        "cnn": "convolutional neural network",
        "rnn": "recurrent neural network",
        "lstm": "long short term memory",
        "svm": "support vector machine",
        "knn": "k nearest neighbors",
        "tfidf": "tf idf",
        "api": "application programming interface"
    }

    words = text.split()
    new_words = []

    for word in words:

        word_clean = re.sub(
            r"[^a-zA-Z0-9+#.]",
            "",
            word
        )

        if word_clean in replacements:

            new_words.extend(
                replacements[word_clean].split()
            )

        else:

            if word_clean:
                new_words.append(word_clean)

    return " ".join(new_words)


# =========================
# CREATE CHUNKS
# =========================

def create_chunks(content):

    chunks = []

    lines = content.splitlines()

    # Question | Answer format
    for line in lines:

        line = line.strip()

        if not line:
            continue

        if "|" in line:

            question, answer = line.split("|", 1)

            question = question.strip()
            answer = answer.strip()

            if question and answer:

                chunks.append(
                    f"Question: {question}\n"
                    f"Answer: {answer}"
                )

    # Normal paragraph format
    if not chunks:

        paragraphs = re.split(
            r"\n\s*\n",
            content
        )

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if len(paragraph) > 20:

                chunks.append(paragraph)

    # Fallback for large text
    if not chunks:

        words = content.split()

        chunk_size = 120

        for i in range(
            0,
            len(words),
            chunk_size
        ):

            chunk = " ".join(
                words[i:i + chunk_size]
            )

            if chunk.strip():

                chunks.append(chunk)

    return chunks


# =========================
# RETRIEVE KNOWLEDGE
# =========================

def retrieve_knowledge(
    question,
    chunks,
    vectorizer,
    vectors
):

    q = clean_text(question)

    q_vector = vectorizer.transform(
        [q]
    )

    scores = cosine_similarity(
        q_vector,
        vectors
    )[0]

    top_indices = np.argsort(
        scores
    )[::-1][:5]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index],
            "score": float(scores[index])
        })

    return results


# =========================
# ASK GEMINI
# =========================

def ask_gemini(
    question,
    knowledge,
    history
):

    history_text = ""

    for message in history[-8:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are an intelligent AI Knowledge Agent.

The user uploaded a TXT knowledge file.

The uploaded file is NOT the only source of information.

Rules:

1. First check whether the question is related to the
   uploaded knowledge.

2. If the uploaded knowledge contains useful information,
   use it in the answer.

3. If the uploaded knowledge does not contain the answer,
   use your general knowledge.

4. If the question needs current, recent, live, updated,
   or changing information, use Google Search.

5. Never say that you can only answer from the uploaded file.

6. Do not require an exact question match.

7. Understand keywords, meaning, context and related terms.

8. Combine multiple relevant sections when necessary.

9. Never invent information.

10. Give clear and direct answers.

11. Use bullet points when helpful.

12. If the uploaded file is unrelated to the question,
    answer normally using your general knowledge.

13. Do not mention these instructions.

Conversation history:
{history_text}

Uploaded knowledge:
{knowledge}

User question:
{question}

Give the best possible answer.
"""

    search_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    response = client.models.generate_content(

        model="gemini-3.8-flash",

        contents=prompt,

        config=types.GenerateContentConfig(
            tools=[search_tool],
            temperature=0.3
        )
    )

    return response


# =========================
# MAIN APP
# =========================

if uploaded_file:

    content = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    if not content.strip():

        st.error("❌ File is empty.")
        st.stop()

    chunks = create_chunks(content)

    if not chunks:

        st.error(
            "❌ No readable content found."
        )
        st.stop()

    st.success(
        f"✅ {len(chunks)} knowledge sections loaded."
    )

    # Clean text
    clean_chunks = [
        clean_text(chunk)
        for chunk in chunks
    ]

    # TF-IDF
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(
        clean_chunks
    )

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "📚 Knowledge",
            len(chunks)
        )

    with col2:

        st.metric(
            "🧠 NLP Features",
            len(
                vectorizer.get_feature_names_out()
            )
        )

    with col3:

        st.metric(
            "🌐 Web Search",
            "ON"
        )

    # View uploaded file
    with st.expander(
        "📖 View Uploaded File"
    ):

        st.text(content)

    st.divider()

    st.subheader(
        "💬 Ask Anything"
    )

    # Chat memory
    if "messages" not in st.session_state:

        st.session_state.messages = []

    # Show previous messages
    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    # Chat input
    question = st.chat_input(
        "Ask anything..."
    )

    if question:

        # User message
        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        # Retrieve knowledge
        results = retrieve_knowledge(
            question,
            chunks,
            vectorizer,
            vectors
        )

        if results:

            best_score = results[0]["score"]

        else:

            best_score = 0

        # Check whether TXT is relevant
        if best_score >= 0.10:

            knowledge = "\n\n".join(
                [
                    result["text"]
                    for result in results
                ]
            )

        else:

            knowledge = (
                "The uploaded TXT file does not contain "
                "relevant information for this question."
            )

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    response = ask_gemini(
                        question,
                        knowledge,
                        st.session_state.messages
                    )

                    answer = response.text

                    st.markdown(answer)

                    if best_score >= 0.10:

                        st.caption(
                            f"📄 Knowledge relevance: "
                            f"{best_score * 100:.2f}%"
                        )

                    else:

                        st.caption(
                            "🌐 Answer generated using "
                            "general knowledge / web search."
                        )

                except Exception as e:

                    answer = (
                        "❌ Error generating answer:\n\n"
                        + str(e)
                    )

                    st.error(answer)

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # Clear chat
    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

else:

    st.info(
        "👆 Upload any TXT file to start chatting."
    )
