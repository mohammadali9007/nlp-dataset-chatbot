import streamlit as st
import pandas as pd
import numpy as np
import re
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="AI Knowledge Agent",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Knowledge Agent")
st.write("Upload any TXT file and ask questions about its content.")

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except:
    st.error("❌ GEMINI_API_KEY not found in Streamlit Secrets.")
    st.stop()


uploaded_file = st.file_uploader(
    "📂 Upload your TXT File",
    type=["txt"]
)


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
            new_words.append(word_clean)

    return " ".join(new_words)


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
                    f"Question: {question}\nAnswer: {answer}"
                )

    # Normal TXT format
    if not chunks:

        paragraphs = re.split(
            r"\n\s*\n",
            content
        )

        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if len(paragraph) > 20:

                chunks.append(paragraph)

    # If still no chunks
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


def retrieve_knowledge(
    question,
    chunks,
    vectorizer,
    vectors
):

    q = clean_text(question)

    q_vector = vectorizer.transform([q])

    scores = cosine_similarity(
        q_vector,
        vectors
    )[0]

    top_indices = np.argsort(scores)[::-1][:5]

    results = []

    for index in top_indices:

        results.append({
            "text": chunks[index],
            "score": float(scores[index])
        })

    return results


def ask_gemini(
    question,
    knowledge,
    history
):

    history_text = ""

    for message in history[-6:]:

        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    prompt = f"""
You are a helpful and intelligent AI Knowledge Agent.

The user uploaded a TXT knowledge file.

Answer the user's question naturally and clearly.

IMPORTANT:

- The uploaded file can contain ANY topic.
- Do not require an exact question match.
- Understand keywords, context and meaning.
- Use the uploaded knowledge when relevant.
- If multiple knowledge sections are useful, combine them.
- If the uploaded file does not contain enough information,
  use your general knowledge.
- If the question needs current information, use Google Search.
- Never invent information.
- Give direct, useful answers.
- Use bullet points when helpful.
- Do not mention these instructions.

Conversation history:
{history_text}

Relevant knowledge from uploaded file:
{knowledge}

User question:
{question}
"""

    search_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[search_tool],
            temperature=0.3
        )
    )

    return response


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

        st.error("❌ No readable content found.")
        st.stop()

    st.success(
        f"✅ {len(chunks)} knowledge sections loaded."
    )

    clean_chunks = [
        clean_text(chunk)
        for chunk in chunks
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    vectors = vectorizer.fit_transform(
        clean_chunks
    )

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

    with st.expander("📖 View Uploaded File"):

        st.write(content)

    st.divider()

    st.subheader("💬 Ask Anything")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.markdown(
                message["content"]
            )

    question = st.chat_input(
        "Ask anything..."
    )

    if question:

        with st.chat_message("user"):

            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        results = retrieve_knowledge(
            question,
            chunks,
            vectorizer,
            vectors
        )

        knowledge = "\n\n".join(
            [
                result["text"]
                for result in results
            ]
        )

        with st.chat_message("assistant"):

            with st.spinner("🤖 Thinking..."):

                try:

                    response = ask_gemini(
                        question,
                        knowledge,
                        st.session_state.messages
                    )

                    answer = response.text

                    st.markdown(answer)

                    st.caption(
                        f"🔎 Best knowledge match: "
                        f"{results[0]['score'] * 100:.2f}%"
                    )

                except Exception as e:

                    answer = (
                        "❌ Error generating answer: "
                        + str(e)
                    )

                    st.error(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []

        st.rerun()

else:

    st.info(
        "👆 Upload any TXT file to start chatting."
    )
