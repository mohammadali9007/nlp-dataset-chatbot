import os
import re
import difflib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)

MODEL_NAME = "gemini-3.8-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "csv_data" not in st.session_state:
    st.session_state.csv_data = {}

if "txt_data" not in st.session_state:
    st.session_state.txt_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    try:

        api_key = None

        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text


# =========================================================
# SIMILARITY
# =========================================================

def similarity(text1, text2):

    return difflib.SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2)
    ).ratio()


# =========================================================
# READ CSV
# =========================================================

def read_csv_file(uploaded_file):

    encodings = [
        "utf-8",
        "latin1",
        "cp1252"
    ]

    for encoding in encodings:

        try:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding=encoding
            )

        except Exception:
            continue

    return None


# =========================================================
# READ TXT
# =========================================================

def read_txt_file(uploaded_file):

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    for encoding in encodings:

        try:

            uploaded_file.seek(0)

            return uploaded_file.read().decode(
                encoding,
                errors="ignore"
            )

        except Exception:
            continue

    return ""


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 IntelliMind AI")

    st.caption(
        "CSV + TXT Intelligent Question Answering"
    )

    st.divider()

    uploaded_files = st.file_uploader(
        "Upload CSV / TXT files",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

    st.divider()

    if st.session_state.csv_data:

        st.subheader("📊 CSV Files")

        for filename, df in st.session_state.csv_data.items():

            st.write(
                f"**{filename}**"
            )

            st.caption(
                f"{len(df):,} rows × "
                f"{len(df.columns)} columns"
            )

    if st.session_state.txt_data:

        st.subheader("📄 TXT Files")

        for filename, text in st.session_state.txt_data.items():

            st.write(
                f"**{filename}**"
            )

            st.caption(
                f"{len(text):,} characters"
            )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# PROCESS FILES
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        # -------------------------
        # CSV
        # -------------------------

        if filename.lower().endswith(".csv"):

            df = read_csv_file(
                uploaded_file
            )

            if df is not None:

                st.session_state.csv_data[
                    filename
                ] = df

        # -------------------------
        # TXT
        # -------------------------

        elif filename.lower().endswith(".txt"):

            text = read_txt_file(
                uploaded_file
            )

            st.session_state.txt_data[
                filename
            ] = text


# =========================================================
# HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.write(
    "Ask questions from your uploaded CSV and TXT files."
)

st.caption(
    "CSV Analysis • TXT Knowledge Base • "
    "Fuzzy Matching • AI Assistant"
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

if st.session_state.csv_data:

    st.subheader("📊 Dataset Overview")

    overview = []

    for filename, df in st.session_state.csv_data.items():

        overview.append({
            "File": filename,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Numeric Columns": len(
                df.select_dtypes(
                    include=np.number
                ).columns
            ),
            "Missing Values": int(
                df.isna().sum().sum()
            )
        })

    overview_df = pd.DataFrame(
        overview
    )

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# CSV FILE DETAILS
# =========================================================

if st.session_state.csv_data:

    with st.expander(
        "📁 View CSV Rows & Columns"
    ):

        for filename, df in st.session_state.csv_data.items():

            st.markdown(
                f"### 📄 {filename}"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Rows",
                    f"{len(df):,}"
                )

            with col2:

                st.metric(
                    "Columns",
                    len(df.columns)
                )

            with col3:

                st.metric(
                    "Missing",
                    int(
                        df.isna().sum().sum()
                    )
                )

            st.write(
                "**Column Names:**"
            )

            st.write(
                ", ".join(
                    map(
                        str,
                        df.columns
                    )
                )
            )

            st.write(
                "**First 10 Rows:**"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.divider()


# =========================================================
# TXT FILE DETAILS
# =========================================================

if st.session_state.txt_data:

    with st.expander(
        "📄 View TXT Information"
    ):

        for filename, text in st.session_state.txt_data.items():

            st.write(
                f"**{filename}**"
            )

            st.write(
                f"Characters: {len(text):,}"
            )

            st.write(
                f"Lines: {len(text.splitlines()):,}"
            )


# =========================================================
# TXT Q&A DATASET SEARCH
# =========================================================

def search_qa_dataset(question):

    best_answer = None
    best_question = None
    best_score = 0
    best_file = None

    normalized_question = normalize_text(
        question
    )

    for filename, text in st.session_state.txt_data.items():

        for line in text.splitlines():

            if "|" not in line:
                continue

            parts = line.split(
                "|",
                1
            )

            if len(parts) != 2:
                continue

            stored_question = parts[0].strip()
            stored_answer = parts[1].strip()

            if not stored_question:
                continue

            if not stored_answer:
                continue

            normalized_stored = normalize_text(
                stored_question
            )

            # Exact match
            if (
                normalized_question
                == normalized_stored
            ):

                return {
                    "answer": stored_answer,
                    "question": stored_question,
                    "score": 1.0,
                    "file": filename
                }

            # Similarity
            sim = similarity(
                question,
                stored_question
            )

            # Word overlap
            q_words = set(
                normalized_question.split()
            )

            s_words = set(
                normalized_stored.split()
            )

            overlap = 0

            if q_words:

                overlap = len(
                    q_words.intersection(
                        s_words
                    )
                ) / len(q_words)

            score = (
                sim * 0.7
                + overlap * 0.3
            )

            if score > best_score:

                best_score = score
                best_answer = stored_answer
                best_question = stored_question
                best_file = filename

    if best_score >= 0.68:

        return {
            "answer": best_answer,
            "question": best_question,
            "score": best_score,
            "file": best_file
        }

    return None


# =========================================================
# LOCAL INTENT ANSWERS
# =========================================================

def local_intent_answer(question):

    q = normalize_text(
        question
    )

    greetings = [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    for item in greetings:

        if (
            q == item
            or similarity(q, item) >= 0.90
        ):

            return (
                "Hello! 👋 "
                "How can I help you today?"
            )

    if (
        similarity(
            q,
            "how are you"
        ) >= 0.80
        or similarity(
            q,
            "how are u"
        ) >= 0.80
    ):

        return (
            "I'm doing great! 😊 "
            "Thanks for asking. "
            "How can I help you?"
        )

    capability_questions = [
        "what can you do",
        "what do you do",
        "what are you doing",
        "what are you do",
        "what can u do"
    ]

    for item in capability_questions:

        if similarity(
            q,
            item
        ) >= 0.78:

            return (
                "I can answer questions from "
                "your uploaded CSV and TXT files, "
                "perform basic data analysis, and "
                "answer general AI and programming questions."
            )

    if similarity(
        q,
        "can you help me"
    ) >= 0.80:

        return (
            "Of course! 😊 "
            "Ask me a question and I'll try to help."
        )

    if (
        similarity(
            q,
            "are you a chatbot"
        ) >= 0.80
    ):

        return (
            "Yes! 🤖 I'm IntelliMind AI, "
            "an AI-powered question-answering chatbot."
        )

    if (
        similarity(
            q,
            "who are you"
        ) >= 0.80
    ):

        return (
            "I'm IntelliMind AI, "
            "an AI-powered question-answering assistant."
        )

    thanks = [
        "thanks",
        "thank you",
        "thank u"
    ]

    for item in thanks:

        if similarity(
            q,
            item
        ) >= 0.85:

            return "You're welcome! 😊"

    goodbye = [
        "goodbye",
        "bye",
        "see you"
    ]

    for item in goodbye:

        if similarity(
            q,
            item
        ) >= 0.85:

            return (
                "Goodbye! 👋 "
                "Have a great day!"
            )

    return None


# =========================================================
# BUILT-IN KNOWLEDGE
# =========================================================

def built_in_answer(question):

    q = normalize_text(
        question
    )

    concepts = {

        "AI": {
            "patterns": [
                "what is ai",
                "ai meaning",
                "ai full form",
                "full form of ai",
                "define ai"
            ],
            "answer": (
                "AI stands for Artificial Intelligence. "
                "AI is a field of computer science that "
                "allows computers to perform tasks that "
                "normally require human intelligence."
            )
        },

        "ML": {
            "patterns": [
                "what is ml",
                "ml meaning",
                "ml full form",
                "full form of ml",
                "define ml",
                "machine learning"
            ],
            "answer": (
                "ML stands for Machine Learning. "
                "Machine Learning is a branch of AI that "
                "allows computers to learn patterns from "
                "data and make predictions or decisions."
            )
        },

        "DL": {
            "patterns": [
                "what is dl",
                "dl meaning",
                "dl full form",
                "full form of dl",
                "define dl",
                "deep learning"
            ],
            "answer": (
                "DL stands for Deep Learning. "
                "Deep Learning is a subfield of Machine "
                "Learning that uses multi-layer neural "
                "networks to learn complex patterns."
            )
        },

        "CV": {
            "patterns": [
                "what is cv",
                "cv meaning",
                "cv full form",
                "full form of cv",
                "define cv",
                "computer vision"
            ],
            "answer": (
                "CV stands for Computer Vision. "
                "Computer Vision is a field of AI that "
                "enables computers to understand and "
                "analyze images and videos."
            )
        },

        "NLP": {
            "patterns": [
                "what is nlp",
                "nlp meaning",
                "nlp full form",
                "full form of nlp",
                "define nlp",
                "natural language processing"
            ],
            "answer": (
                "NLP stands for Natural Language Processing. "
                "NLP is a branch of AI that enables computers "
                "to process, understand, and generate human language."
            )
        },

        "Python": {
            "patterns": [
                "what is python",
                "python meaning",
                "define python"
            ],
            "answer": (
                "Python is a high-level programming language "
                "widely used in web development, data science, "
                "machine learning, AI, and automation."
            )
        }
    }

    for concept, data in concepts.items():

        for pattern in data["patterns"]:

            if q == normalize_text(
                pattern
            ):

                return data["answer"]

            if similarity(
                q,
                pattern
            ) >= 0.82:

                return data["answer"]

    # Short forms
    short_forms = {

        "ai":
            "AI stands for Artificial Intelligence.",

        "ml":
            "ML stands for Machine Learning.",

        "dl":
            "DL stands for Deep Learning.",

        "cv":
            "CV stands for Computer Vision.",

        "nlp":
            "NLP stands for Natural Language Processing."
    }

    if q in short_forms:

        return short_forms[q]

    return None


# =========================================================
# COLUMN SYNONYMS
# =========================================================

COLUMN_SYNONYMS = {

    "glucose": [
        "glucose",
        "blood sugar",
        "blood glucose",
        "sugar",
        "glucose level",
        "blood sugar level"
    ],

    "bloodpressure": [
        "blood pressure",
        "pressure",
        "bp"
    ],

    "pregnancies": [
        "pregnancy",
        "pregnancies",
        "pregnant"
    ],

    "bmi": [
        "bmi",
        "body mass index"
    ],

    "insulin": [
        "insulin"
    ],

    "age": [
        "age",
        "ages"
    ],

    "skinthickness": [
        "skin thickness",
        "skin"
    ],

    "diabetespedigreefunction": [
        "diabetes pedigree",
        "pedigree"
    ],

    "outcome": [
        "outcome",
        "diabetes",
        "diabetic",
        "diabetes outcome"
    ],

    "sc": [
        "creatinine",
        "serum creatinine",
        "sc"
    ],

    "bu": [
        "urea",
        "blood urea",
        "bu"
    ],

    "hemo": [
        "hemoglobin",
        "haemoglobin",
        "hemo"
    ],

    "sod": [
        "sodium",
        "sod"
    ],

    "pot": [
        "potassium",
        "pot"
    ],

    "classification": [
        "classification",
        "class",
        "result"
    ]
}


# =========================================================
# FIND COLUMNS
# =========================================================

def find_relevant_columns(question):

    q = normalize_text(
        question
    )

    all_columns = []

    for filename, df in st.session_state.csv_data.items():

        for col in df.columns:

            if str(col) not in all_columns:

                all_columns.append(
                    str(col)
                )

    found = []

    # Direct match
    for col in all_columns:

        col_normalized = normalize_text(
            col
        )

        if col_normalized in q:

            found.append(col)

    # Synonyms
    for actual_column, words in COLUMN_SYNONYMS.items():

        real_column = None

        for col in all_columns:

            if normalize_text(col) == actual_column:

                real_column = col

                break

        if real_column is None:
            continue

        for word in words:

            if normalize_text(word) in q:

                if real_column not in found:

                    found.append(
                        real_column
                    )

                break

    return found


# =========================================================
# FIND BEST CSV
# =========================================================

def find_best_csv(question):

    if not st.session_state.csv_data:

        return None, None

    relevant = find_relevant_columns(
        question
    )

    if relevant:

        for filename, df in st.session_state.csv_data.items():

            for col in relevant:

                if col in df.columns:

                    return filename, df

    filename = list(
        st.session_state.csv_data.keys()
    )[0]

    return (
        filename,
        st.session_state.csv_data[
            filename
        ]
    )


# =========================================================
# DIRECT CSV ANALYSIS
# =========================================================

def direct_csv_analysis(question):

    if not st.session_state.csv_data:

        return None

    q = normalize_text(
        question
    )

    filename, df = find_best_csv(
        question
    )

    if df is None:

        return None

    relevant = find_relevant_columns(
        question
    )


    # =====================================================
    # ROWS
    # =====================================================

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "how many records" in q
        or "total records" in q
        or "how many patients" in q
        or "total patients" in q
    ):

        return (
            f"The dataset contains "
            f"**{len(df):,} rows**."
        )


    # =====================================================
    # COLUMNS
    # =====================================================

    if (
        "how many columns" in q
        or "number of columns" in q
        or "total columns" in q
        or "how many fields" in q
    ):

        return (
            f"The dataset contains "
            f"**{len(df.columns)} columns**."
        )


    # =====================================================
    # COLUMN NAMES
    # =====================================================

    if (
        "column names" in q
        or "what are the columns" in q
        or "list columns" in q
        or "show columns" in q
    ):

        columns = ", ".join(
            map(
                str,
                df.columns
            )
        )

        return (
            f"The columns are:\n\n"
            f"{columns}"
        )


    # =====================================================
    # DATASET SHAPE
    # =====================================================

    if (
        "shape" in q
        or "dataset size" in q
        or "size of dataset" in q
    ):

        return (
            f"The dataset has "
            f"**{len(df):,} rows** and "
            f"**{len(df.columns)} columns**."
        )


    # =====================================================
    # MISSING VALUES
    # =====================================================

    if (
        "missing" in q
        or "null" in q
        or "empty values" in q
    ):

        missing = int(
            df.isna().sum().sum()
        )

        return (
            f"The dataset contains "
            f"**{missing:,} missing values**."
        )


    # =====================================================
    # UNIQUE
    # =====================================================

    if "unique values" in q:

        if relevant:

            col = relevant[0]

            if col in df.columns:

                count = df[col].nunique(
                    dropna=True
                )

                return (
                    f"**{col}** has "
                    f"**{count:,} unique values**."
                )


    # =====================================================
    # DIABETIC COUNT
    # =====================================================

    if (
        "how many diabetic" in q
        or "number of diabetic" in q
        or "diabetic patients" in q
        or "diabetes patients" in q
    ):

        outcome_col = None

        for col in df.columns:

            if normalize_text(col) == "outcome":

                outcome_col = col

                break

        if outcome_col:

            values = pd.to_numeric(
                df[outcome_col],
                errors="coerce"
            )

            count = int(
                (values == 1).sum()
            )

            return (
                f"There are **{count:,} "
                f"diabetic patients** "
                f"in the dataset."
            )


    # =====================================================
    # AVERAGE
    # =====================================================

    if (
        "average" in q
        or "avg" in q
        or "mean" in q
        or "avrage" in q
        or "averge" in q
    ):

        if relevant:

            col = relevant[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    avg = values.mean()

                    return (
                        f"The average **{col}** "
                        f"is **{avg:.2f}**."
                    )


    # =====================================================
    # MAXIMUM
    # =====================================================

    if (
        "highest" in q
        or "higest" in q
        or "maximum" in q
        or "largest" in q
        or "greatest" in q
    ):

        if relevant:

            col = relevant[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    maximum = values.max()

                    return (
                        f"The highest **{col}** "
                        f"value is **{maximum}**."
                    )


    # =====================================================
    # MINIMUM
    # =====================================================

    if (
        "lowest" in q
        or "lowset" in q
        or "minimum" in q
        or "smallest" in q
    ):

        if relevant:

            col = relevant[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    minimum = values.min()

                    return (
                        f"The lowest **{col}** "
                        f"value is **{minimum}**."
                    )


    # =====================================================
    # SUM
    # =====================================================

    if (
        "sum of" in q
        or "total of" in q
        or "sum" in q
    ):

        if relevant:

            col = relevant[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    total = values.sum()

                    return (
                        f"The total **{col}** "
                        f"is **{total:,.2f}**."
                    )


    # =====================================================
    # MEDIAN
    # =====================================================

    if "median" in q:

        if relevant:

            col = relevant[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    median = values.median()

                    return (
                        f"The median **{col}** "
                        f"is **{median:.2f}**."
                    )

    return None


# =========================================================
# TXT SEARCH
# =========================================================

def split_text(
    text,
    chunk_size=1200
):

    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    chunks = []

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if (
            len(current)
            + len(paragraph)
            <= chunk_size
        ):

            current += (
                "\n" + paragraph
            )

        else:

            if current.strip():

                chunks.append(
                    current.strip()
                )

            current = paragraph

    if current.strip():

        chunks.append(
            current.strip()
        )

    return chunks


# =========================================================
# SEARCH TXT
# =========================================================

def search_txt(
    question,
    top_k=5
):

    if not st.session_state.txt_data:

        return []

    documents = []
    filenames = []

    for filename, text in st.session_state.txt_data.items():

        chunks = split_text(
            text
        )

        for chunk in chunks:

            documents.append(
                chunk
            )

            filenames.append(
                filename
            )

    if not documents:

        return []

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        question_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            question_vector,
            matrix
        )[0]

        indexes = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in indexes:

            if scores[idx] > 0.05:

                results.append({
                    "text": documents[idx],
                    "score": float(
                        scores[idx]
                    ),
                    "file": filenames[idx]
                })

        return results

    except Exception:

        return []


# =========================================================
# BUILD CSV CONTEXT
# =========================================================

def build_csv_context(question):

    if not st.session_state.csv_data:

        return ""

    context = []

    for filename, df in st.session_state.csv_data.items():

        relevant = find_relevant_columns(
            question
        )

        valid = [
            col
            for col in relevant
            if col in df.columns
        ]

        if valid:

            sample = df[
                valid
            ].head(15)

        else:

            sample = df.head(10)

        context.append(
            f"""
FILE: {filename}

ROWS: {len(df)}

COLUMNS:
{list(df.columns)}

DATA:
{sample.to_string(index=False)}
"""
        )

    return "\n".join(
        context
    )


# =========================================================
# BUILD TXT CONTEXT
# =========================================================

def build_txt_context(question):

    results = search_txt(
        question
    )

    if not results:

        return ""

    context = []

    for item in results:

        context.append(
            f"""
FILE: {item['file']}

CONTENT:
{item['text']}
"""
        )

    return "\n".join(
        context
    )


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    csv_context,
    txt_context
):

    client = get_gemini_client()

    if client is None:

        return None, "API key not configured"

    prompt = f"""
You are IntelliMind AI.

Answer the user's question clearly and naturally.

USER QUESTION:
{question}

RULES:

1. Use uploaded CSV data when the question is about CSV.
2. Never invent CSV numbers.
3. If numerical analysis is required, use the actual data.
4. Use TXT information when relevant.
5. If the question is a general educational question,
   answer normally.
6. Keep answers concise but useful.
7. Do not mention internal processing.
8. Do not show column evidence unless the user explicitly asks.
9. Do not say that you cannot access the uploaded files.

CSV DATA:
{csv_context if csv_context else "No CSV uploaded."}

TXT DATA:
{txt_context if txt_context else "No TXT information found."}

Now answer the user.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return (
                response.text.strip(),
                None
            )

        return None, "Empty response"

    except Exception as e:

        return None, str(e)


# =========================================================
# SHOW CHAT HISTORY
# =========================================================

for chat in st.session_state.chat_history:

    with st.chat_message(
        chat["role"]
    ):

        st.markdown(
            chat["content"]
        )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask me anything about your files..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    # -----------------------------------------------------
    # USER QUESTION
    # -----------------------------------------------------

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =====================================================
    # 1. CSV DIRECT ANSWER
    # =====================================================

    csv_answer = direct_csv_analysis(
        question
    )

    if csv_answer:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": csv_answer
        })

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                csv_answer
            )

        st.stop()


    # =====================================================
    # 2. TXT Q&A
    # =====================================================

    qa_result = search_qa_dataset(
        question
    )

    if qa_result:

        answer = qa_result["answer"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                answer
            )

        st.stop()


    # =====================================================
    # 3. LOCAL INTENT
    # =====================================================

    local_answer = local_intent_answer(
        question
    )

    if local_answer:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": local_answer
        })

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                local_answer
            )

        st.stop()


    # =====================================================
    # 4. BUILT-IN KNOWLEDGE
    # =====================================================

    builtin_answer = built_in_answer(
        question
    )

    if builtin_answer:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": builtin_answer
        })

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                builtin_answer
            )

        st.stop()


    # =====================================================
    # 5. GEMINI AI
    # =====================================================

    csv_context = build_csv_context(
        question
    )

    txt_context = build_txt_context(
        question
    )

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "🤔 Thinking..."
        ):

            answer, error = generate_ai_answer(
                question,
                csv_context,
                txt_context
            )

        if answer:

            st.markdown(
                answer
            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer
            })

        else:

            fallback = (
                "Sorry, I couldn't generate an answer "
                "right now. Please try again."
            )

            st.warning(
                fallback
            )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": fallback
            })
