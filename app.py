import streamlit as st
import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="NLP AI Agent",
    page_icon="🤖",
    layout="centered"
)


# ==========================================
# TITLE
# ==========================================

st.title("🤖 NLP AI Knowledge Agent")

st.write(
    "Upload a TXT Q&A dataset and ask questions "
    "related to the uploaded knowledge."
)


# ==========================================
# DATASET UPLOAD
# ==========================================

uploaded_file = st.file_uploader(
    "📂 Upload your TXT Dataset",
    type=["txt"]
)


# ==========================================
# MAIN SYSTEM
# ==========================================

if uploaded_file is not None:

    questions = []
    answers = []

    # Read uploaded file
    content = uploaded_file.read().decode(
        "utf-8",
        errors="ignore"
    )

    # Extract Question and Answer
    for line in content.splitlines():

        line = line.strip()

        if not line:
            continue

        if "|" in line:

            question, answer = line.split(
                "|",
                1
            )

            questions.append(
                question.strip()
            )

            answers.append(
                answer.strip()
            )


    # ==========================================
    # CHECK DATASET
    # ==========================================

    if len(questions) == 0:

        st.error(
            "❌ No valid Question | Answer data found."
        )

        st.stop()


    st.success(
        f"✅ Dataset uploaded successfully! "
        f"{len(questions)} questions loaded."
    )


    # ==========================================
    # TEXT PREPROCESSING
    # ==========================================

    stop_words = {
        "the", "is", "a", "an", "of",
        "to", "in", "on", "for",
        "and", "or", "what", "how",
        "why", "are", "was", "were",
        "can", "does", "do",
        "where", "when", "which"
    }


    def clean_text(text):

        text = text.lower()


        # AI / ML / NLP abbreviations

        replacements = {

            "ai":
            "artificial intelligence",

            "ml":
            "machine learning",

            "dl":
            "deep learning",

            "nlp":
            "natural language processing",

            "llm":
            "large language model",

            "cv":
            "computer vision",

            "cnn":
            "convolutional neural network",

            "rnn":
            "recurrent neural network",

            "lstm":
            "long short term memory",

            "svm":
            "support vector machine",

            "knn":
            "k nearest neighbors",

            "tfidf":
            "tf idf",

            "api":
            "application programming interface"
        }


        words = text.split()

        new_words = []


        for word in words:

            word_clean = re.sub(
                r"[^a-zA-Z0-9]",
                "",
                word
            )


            if word_clean in replacements:

                new_words.extend(
                    replacements[
                        word_clean
                    ].split()
                )

            else:

                new_words.append(
                    word_clean
                )


        text = " ".join(
            new_words
        )


        words = text.split()


        words = [

            word

            for word in words

            if word not in stop_words

        ]


        return " ".join(words)


    # ==========================================
    # PROCESS QUESTIONS
    # ==========================================

    clean_questions = [

        clean_text(q)

        for q in questions

    ]


    # ==========================================
    # TF-IDF MODEL
    # ==========================================

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True
    )


    question_vectors = vectorizer.fit_transform(
        clean_questions
    )


    # ==========================================
    # CHATBOT FUNCTION
    # ==========================================

    def chatbot(user_question):

        cleaned_question = clean_text(
            user_question
        )


        user_vector = vectorizer.transform(
            [cleaned_question]
        )


        similarity = cosine_similarity(
            user_vector,
            question_vectors
        )


        best_index = np.argmax(
            similarity
        )


        best_score = similarity[
            0
        ][best_index]


        # Confidence threshold

        if best_score < 0.20:

            return (
                "Sorry, I couldn't find a reliable "
                "answer from the uploaded dataset.",
                best_score
            )


        return (
            answers[best_index],
            best_score
        )


    # ==========================================
    # DATASET INFORMATION
    # ==========================================

    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "📚 Questions",
            len(questions)
        )


    with col2:

        st.metric(
            "🧠 NLP Features",
            len(
                vectorizer
                .get_feature_names_out()
            )
        )


    # ==========================================
    # SHOW DATASET
    # ==========================================

    with st.expander(
        "📖 View Uploaded Dataset"
    ):

        df = pd.DataFrame({

            "Question": questions,

            "Answer": answers

        })


        st.dataframe(
            df,
            use_container_width=True
        )


    st.divider()


    # ==========================================
    # CHAT
    # ==========================================

    st.subheader(
        "💬 Ask Your Question"
    )


    if "messages" not in st.session_state:

        st.session_state.messages = []


    # Display chat history

    for message in st.session_state.messages:

        with st.chat_message(
            message["role"]
        ):

            st.write(
                message["content"]
            )


            if (
                message["role"] == "assistant"
                and "score" in message
            ):

                st.caption(
                    "Similarity: "
                    + str(
                        round(
                            message["score"] * 100,
                            2
                        )
                    )
                    + "%"
                )


    # ==========================================
    # USER QUESTION
    # ==========================================

    user_question = st.chat_input(
        "Ask about AI, ML, NLP, DL, LLM..."
    )


    if user_question:

        # User message

        st.session_state.messages.append({

            "role": "user",

            "content": user_question

        })


        with st.chat_message(
            "user"
        ):

            st.write(
                user_question
            )


        # Generate answer

        answer, score = chatbot(
            user_question
        )


        # Bot message

        st.session_state.messages.append({

            "role": "assistant",

            "content": answer,

            "score": score

        })


        with st.chat_message(
            "assistant"
        ):

            st.write(
                answer
            )


            st.caption(
                "Similarity: "
                + str(
                    round(
                        score * 100,
                        2
                    )
                )
                + "%"
            )


    # ==========================================
    # CLEAR CHAT
    # ==========================================

    if st.button(
        "🗑️ Clear Chat"
    ):

        st.session_state.messages = []

        st.rerun()


else:

    st.info(
        "👆 Upload a TXT dataset to start chatting."
    )


    st.markdown("""
### 📌 Dataset Format

Your TXT file should contain:

`Question | Answer`

Example:

`What is AI? | Artificial Intelligence is a field of computer science.`

`How does Machine Learning work? | Machine Learning learns patterns from data.`

`What is NLP? | NLP helps computers understand human language.`

### 🧠 Supported Topics

- Artificial Intelligence
- Machine Learning
- Deep Learning
- NLP
- LLM
- AI Agents
- Computer Vision
- Neural Networks
- Python
- Data Science
""")
