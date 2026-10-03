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
# GEMINI CONFIGURATION
# =========================================================

api_key = None

try:
    api_key = st.secrets.get("GEMINI_API_KEY", None)
except Exception:
    api_key = None

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:
    try:
        client = genai.Client(
            api_key=api_key
        )
    except Exception:
        client = None


# =========================================================
# HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.caption(
    "AI Question Answering Assistant • "
    "CSV Analysis • TXT Knowledge Base"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload CSV or TXT files",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            try:

                # =========================================
                # CSV
                # =========================================

                if file.name.lower().endswith(".csv"):

                    try:

                        df = pd.read_csv(
                            file
                        )

                    except Exception:

                        file.seek(0)

                        try:

                            df = pd.read_csv(
                                file,
                                encoding="latin1"
                            )

                        except Exception:

                            file.seek(0)

                            df = pd.read_csv(
                                file,
                                encoding="cp1252"
                            )

                    st.session_state.csv_data[
                        file.name
                    ] = df

                # =========================================
                # TXT
                # =========================================

                elif file.name.lower().endswith(".txt"):

                    raw = file.getvalue()

                    try:

                        text = raw.decode(
                            "utf-8"
                        )

                    except UnicodeDecodeError:

                        try:

                            text = raw.decode(
                                "utf-8-sig"
                            )

                        except UnicodeDecodeError:

                            try:

                                text = raw.decode(
                                    "cp1252"
                                )

                            except UnicodeDecodeError:

                                text = raw.decode(
                                    "latin1",
                                    errors="ignore"
                                )

                    st.session_state.txt_data[
                        file.name
                    ] = text

            except Exception as e:

                st.error(
                    f"Error reading {file.name}: {e}"
                )

    st.divider()

    # =====================================================
    # CSV FILES
    # =====================================================

    st.subheader("📊 CSV Files")

    if st.session_state.csv_data:

        for name, df in st.session_state.csv_data.items():

            st.write(
                f"**{name}**"
            )

            st.caption(
                f"Rows: {len(df):,} | "
                f"Columns: {len(df.columns)}"
            )

    else:

        st.info(
            "No CSV file uploaded."
        )

    st.divider()

    # =====================================================
    # TXT FILES
    # =====================================================

    st.subheader("📄 TXT Files")

    if st.session_state.txt_data:

        for name, text in st.session_state.txt_data.items():

            st.write(
                f"**{name}**"
            )

            st.caption(
                f"Characters: {len(text):,}"
            )

    else:

        st.info(
            "No TXT file uploaded."
        )

    st.divider()

    # =====================================================
    # CLEAR CHAT
    # =====================================================

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace(
        "_",
        " "
    )

    text = text.replace(
        "-",
        " "
    )

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

    return text.strip()


# =========================================================
# FUZZY SIMILARITY
# =========================================================

def similarity(
    text1,
    text2
):

    text1 = normalize_text(
        text1
    )

    text2 = normalize_text(
        text2
    )

    if not text1 or not text2:
        return 0

    return difflib.SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


# =========================================================
# FUZZY MATCH WORD
# =========================================================

def fuzzy_match(
    word,
    choices,
    cutoff=0.65
):

    if not choices:
        return None

    word = normalize_text(
        word
    )

    normalized = {
        normalize_text(c): c
        for c in choices
    }

    matches = difflib.get_close_matches(
        word,
        normalized.keys(),
        n=1,
        cutoff=cutoff
    )

    if matches:

        return normalized[
            matches[0]
        ]

    return None


# =========================================================
# Q&A DATASET SEARCH
# =========================================================

def search_qa_dataset(
    question
):

    question_normalized = normalize_text(
        question
    )

    best_answer = None
    best_question = None
    best_score = 0
    best_file = None

    for filename, text in st.session_state.txt_data.items():

        for line in text.splitlines():

            line = line.strip()

            if not line:
                continue

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

            if (
                not stored_question
                or not stored_answer
            ):
                continue

            stored_normalized = normalize_text(
                stored_question
            )

            # =============================================
            # EXACT MATCH
            # =============================================

            if (
                question_normalized
                == stored_normalized
            ):

                return {
                    "answer": stored_answer,
                    "matched_question": stored_question,
                    "score": 1.0,
                    "file": filename
                }

            # =============================================
            # FUZZY MATCH
            # =============================================

            score = similarity(
                question_normalized,
                stored_normalized
            )

            # =============================================
            # WORD OVERLAP
            # =============================================

            q_words = set(
                question_normalized.split()
            )

            s_words = set(
                stored_normalized.split()
            )

            if q_words and s_words:

                overlap = len(
                    q_words & s_words
                ) / max(
                    len(q_words),
                    len(s_words)
                )

            else:

                overlap = 0

            final_score = (
                score * 0.7
                + overlap * 0.3
            )

            if final_score > best_score:

                best_score = final_score
                best_answer = stored_answer
                best_question = stored_question
                best_file = filename

    # =============================================
    # ACCEPT FUZZY MATCH
    # =============================================

    if best_score >= 0.68:

        return {
            "answer": best_answer,
            "matched_question": best_question,
            "score": best_score,
            "file": best_file
        }

    return None


# =========================================================
# INTENT BASED LOCAL ANSWERS
# =========================================================

def local_intent_answer(
    question
):

    q = normalize_text(
        question
    )

    # =====================================================
    # GREETING
    # =====================================================

    greetings = [
        "hi",
        "hello",
        "hey",
        "hi there",
        "hello there",
        "hey there",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    for greeting in greetings:

        if q == greeting:

            if greeting == "hi":
                return "Hello! How can I help you?"

            if greeting == "hello":
                return (
                    "Hello! How can I help you today?"
                )

            if greeting == "hey":
                return (
                    "Hey! How can I help you?"
                )

            if greeting == "hi there":
                return (
                    "Hi there! What would you like to know?"
                )

            if greeting == "hello there":
                return (
                    "Hello! Feel free to ask me a question."
                )

            if greeting == "good morning":
                return (
                    "Good morning! How can I help you?"
                )

            if greeting == "good afternoon":
                return (
                    "Good afternoon! How can I help you?"
                )

            if greeting == "good evening":
                return (
                    "Good evening! What can I help you with?"
                )

    # =====================================================
    # HOW ARE YOU
    # =====================================================

    how_are_you_patterns = [
        "how are you",
        "how are you doing",
        "how r you",
        "how r u"
    ]

    for pattern in how_are_you_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.78:

            return (
                "I'm doing great! "
                "Thanks for asking. "
                "How can I help you?"
            )

    # =====================================================
    # NAME
    # =====================================================

    name_patterns = [
        "what is your name",
        "whats your name",
        "who are you"
    ]

    for pattern in name_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.72:

            return (
                "I'm IntelliMind AI, "
                "an AI-powered question answering chatbot."
            )

    # =====================================================
    # WHAT CAN YOU DO
    # =====================================================

    capability_patterns = [
        "what can you do",
        "what do you do",
        "what are you doing",
        "what are you do",
        "what can u do",
        "what you can do"
    ]

    for pattern in capability_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.65:

            return (
                "I can answer questions using "
                "your uploaded CSV and TXT knowledge base. "
                "I can also analyze numerical data such as "
                "average, highest, lowest, total, missing "
                "values, and patient records."
            )

    # =====================================================
    # HELP
    # =====================================================

    help_patterns = [
        "can you help me",
        "can u help me",
        "help me",
        "are you able to help"
    ]

    for pattern in help_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "Of course! Ask me a question and "
                "I'll try to help using the available "
                "knowledge base."
            )

    # =====================================================
    # CHATBOT
    # =====================================================

    chatbot_patterns = [
        "are you a chatbot",
        "are you chatbot",
        "you are chatbot",
        "is this a chatbot"
    ]

    for pattern in chatbot_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "Yes. I'm an AI-powered question "
                "answering chatbot."
            )

    # =====================================================
    # THANK YOU
    # =====================================================

    thanks_patterns = [
        "thank you",
        "thanks",
        "thank you so much",
        "thanks a lot"
    ]

    for pattern in thanks_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "You're welcome! "
                "I'm happy to help."
            )

    # =====================================================
    # GOODBYE
    # =====================================================

    goodbye_patterns = [
        "goodbye",
        "bye",
        "see you later",
        "see you"
    ]

    for pattern in goodbye_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.72:

            return (
                "Goodbye! Have a great day!"
            )

    return None


# =========================================================
# BUILT-IN AI ANSWERS
# =========================================================

def built_in_answer(
    question
):

    q = normalize_text(
        question
    )

    # =====================================================
    # MACHINE LEARNING
    # =====================================================

    ml_patterns = [
        "what is ml",
        "what is machine learning",
        "define machine learning",
        "machine learning meaning",
        "ml meaning"
    ]

    for pattern in ml_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "**Machine Learning (ML)** is a branch "
                "of Artificial Intelligence (AI) that allows "
                "computers to learn patterns from data and "
                "make predictions or decisions without "
                "being explicitly programmed for every task.\n\n"
                "**Example:** A machine learning model can "
                "learn from previous patient data and predict "
                "whether a patient may have diabetes."
            )

    # =====================================================
    # ARTIFICIAL INTELLIGENCE
    # =====================================================

    ai_patterns = [
        "what is ai",
        "what is artificial intelligence",
        "define artificial intelligence"
    ]

    for pattern in ai_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "**Artificial Intelligence (AI)** is a "
                "field of computer science that focuses "
                "on creating systems that can perform tasks "
                "that normally require human intelligence, "
                "such as learning, reasoning, understanding "
                "language, and recognizing patterns."
            )

    # =====================================================
    # DEEP LEARNING
    # =====================================================

    dl_patterns = [
        "what is dl",
        "what is deep learning",
        "define deep learning"
    ]

    for pattern in dl_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "**Deep Learning (DL)** is a subfield "
                "of Machine Learning that uses multi-layer "
                "neural networks to learn complex patterns "
                "from large amounts of data."
            )

    # =====================================================
    # NLP
    # =====================================================

    nlp_patterns = [
        "what is nlp",
        "what is natural language processing",
        "define nlp"
    ]

    for pattern in nlp_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "**Natural Language Processing (NLP)** "
                "is a branch of AI that enables computers "
                "to process, understand, and generate "
                "human language.\n\n"
                "Examples include chatbots, translation, "
                "sentiment analysis, text classification, "
                "and question answering."
            )

    # =====================================================
    # PYTHON
    # =====================================================

    python_patterns = [
        "what is python",
        "define python",
        "python meaning"
    ]

    for pattern in python_patterns:

        if similarity(
            q,
            pattern
        ) >= 0.70:

            return (
                "**Python** is a high-level programming "
                "language known for its simple syntax and "
                "wide use in web development, data science, "
                "Machine Learning, AI, and automation."
            )

    return None


# =========================================================
# CSV COLUMN LIST
# =========================================================

def get_all_columns():

    columns = []

    for df in st.session_state.csv_data.values():

        for col in df.columns:

            if col not in columns:

                columns.append(col)

    return columns


# =========================================================
# NUMERIC COLUMNS
# =========================================================

def get_numeric_columns():

    columns = []

    for df in st.session_state.csv_data.values():

        for col in df.columns:

            if pd.api.types.is_numeric_dtype(
                df[col]
            ):

                if col not in columns:

                    columns.append(col)

    return columns


# =========================================================
# FIND RELEVANT CSV COLUMNS
# =========================================================

def find_relevant_columns(
    question
):

    q = normalize_text(
        question
    )

    all_columns = get_all_columns()

    matched = []

    # =====================================================
    # DIRECT COLUMN MATCH
    # =====================================================

    for column in all_columns:

        normalized_column = normalize_text(
            column
        )

        if normalized_column in q:

            if column not in matched:

                matched.append(
                    column
                )

            continue

        # Match individual words
        for word in normalized_column.split():

            if len(word) < 3:
                continue

            if word in q:

                if column not in matched:

                    matched.append(
                        column
                    )

                break

    # =====================================================
    # SYNONYMS
    # =====================================================

    synonyms = {

        "sugar": [
            "Glucose"
        ],

        "blood sugar": [
            "Glucose"
        ],

        "blood glucose": [
            "Glucose"
        ],

        "glucose level": [
            "Glucose"
        ],

        "pressure": [
            "BloodPressure",
            "bp"
        ],

        "blood pressure": [
            "BloodPressure",
            "bp"
        ],

        "pregnancy": [
            "Pregnancies"
        ],

        "pregnancies": [
            "Pregnancies"
        ],

        "pregnant": [
            "Pregnancies"
        ],

        "diabetes": [
            "Outcome",
            "dm"
        ],

        "diabetic": [
            "Outcome",
            "dm"
        ],

        "bmi": [
            "BMI"
        ],

        "insulin": [
            "Insulin"
        ],

        "age": [
            "Age",
            "age"
        ],

        "skin": [
            "SkinThickness"
        ],

        "pedigree": [
            "DiabetesPedigreeFunction"
        ],

        "creatinine": [
            "sc"
        ],

        "urea": [
            "bu"
        ],

        "hemoglobin": [
            "hemo"
        ],

        "sodium": [
            "sod"
        ],

        "potassium": [
            "pot"
        ],

        "classification": [
            "classification"
        ]
    }

    for keyword, possible_columns in synonyms.items():

        if keyword in q:

            for possible_column in possible_columns:

                for actual_column in all_columns:

                    if normalize_text(
                        actual_column
                    ) == normalize_text(
                        possible_column
                    ):

                        if actual_column not in matched:

                            matched.append(
                                actual_column
                            )

    return matched


# =========================================================
# CSV DIRECT ANALYSIS
# =========================================================

def direct_analysis(
    question
):

    q = normalize_text(
        question
    )

    relevant_columns = find_relevant_columns(
        question
    )

    # =====================================================
    # TOTAL PATIENTS / ROWS
    # =====================================================

    if any(
        x in q
        for x in [
            "how many patients",
            "number of patients",
            "total patients",
            "how many rows",
            "number of rows",
            "total rows",
            "how many records",
            "number of records",
            "total records"
        ]
    ):

        # ---------------------------------------------
        # DIABETIC PATIENTS
        # ---------------------------------------------

        if (
            "diabetic" in q
            or "diabetes patients" in q
            or "diabetes patient" in q
        ):

            count = 0

            for df in st.session_state.csv_data.values():

                outcome_column = None

                for col in df.columns:

                    if normalize_text(
                        col
                    ) == "outcome":

                        outcome_column = col

                        break

                if outcome_column:

                    values = pd.to_numeric(
                        df[outcome_column],
                        errors="coerce"
                    )

                    count += int(
                        (values == 1).sum()
                    )

            if count > 0:

                return (
                    f"There are **{count:,} diabetic "
                    f"patients** in the uploaded dataset."
                )

        # ---------------------------------------------
        # TOTAL ROWS
        # ---------------------------------------------

        total = sum(
            len(df)
            for df in st.session_state.csv_data.values()
        )

        return (
            f"There are **{total:,} total records** "
            f"in the uploaded CSV files."
        )

    # =====================================================
    # COLUMN COUNT
    # =====================================================

    if any(
        x in q
        for x in [
            "how many columns",
            "number of columns",
            "total columns"
        ]
    ):

        total = sum(
            len(df.columns)
            for df in st.session_state.csv_data.values()
        )

        return (
            f"There are **{total} columns** "
            f"in the uploaded CSV files."
        )

    # =====================================================
    # COLUMN NAMES
    # =====================================================

    if any(
        x in q
        for x in [
            "column names",
            "what columns",
            "list columns",
            "show columns"
        ]
    ):

        result = []

        for filename, df in st.session_state.csv_data.items():

            result.append(
                f"**{filename}:**\n"
                + ", ".join(
                    str(c)
                    for c in df.columns
                )
            )

        return "\n\n".join(
            result
        )

    # =====================================================
    # MISSING VALUES
    # =====================================================

    if any(
        x in q
        for x in [
            "missing",
            "missing values",
            "null",
            "null values",
            "empty values"
        ]
    ):

        output = []

        for filename, df in st.session_state.csv_data.items():

            missing = df.isnull().sum()

            missing = missing[
                missing > 0
            ]

            if len(missing) == 0:

                output.append(
                    f"**{filename}:** "
                    "No missing values."
                )

            else:

                lines = []

                for col, count in missing.items():

                    lines.append(
                        f"{col}: {int(count)}"
                    )

                output.append(
                    f"**{filename}:**\n"
                    + "\n".join(lines)
                )

        return "\n\n".join(
            output
        )

    # =====================================================
    # AVERAGE
    # =====================================================

    if any(
        x in q
        for x in [
            "average",
            "avg",
            "mean",
            "avrage"
        ]
    ):

        if not relevant_columns:

            relevant_columns = get_numeric_columns()

        output = []

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    continue

                value = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).mean()

                if pd.notna(value):

                    output.append(
                        f"**{col}** average: "
                        f"**{value:.2f}** "
                        f"({filename})"
                    )

        if output:

            return "\n".join(
                output
            )

    # =====================================================
    # HIGHEST
    # =====================================================

    if any(
        x in q
        for x in [
            "highest",
            "higest",
            "maximum",
            "max",
            "largest",
            "greatest"
        ]
    ):

        if not relevant_columns:

            relevant_columns = get_numeric_columns()

        output = []

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    continue

                value = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).max()

                if pd.notna(value):

                    output.append(
                        f"**{col}** highest value: "
                        f"**{value:g}** "
                        f"({filename})"
                    )

        if output:

            return "\n".join(
                output
            )

    # =====================================================
    # LOWEST
    # =====================================================

    if any(
        x in q
        for x in [
            "lowest",
            "lowset",
            "minimum",
            "min",
            "smallest"
        ]
    ):

        if not relevant_columns:

            relevant_columns = get_numeric_columns()

        output = []

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    continue

                value = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).min()

                if pd.notna(value):

                    output.append(
                        f"**{col}** lowest value: "
                        f"**{value:g}** "
                        f"({filename})"
                    )

        if output:

            return "\n".join(
                output
            )

    # =====================================================
    # SUM
    # =====================================================

    if any(
        x in q
        for x in [
            "sum",
            "total of"
        ]
    ):

        if not relevant_columns:

            relevant_columns = get_numeric_columns()

        output = []

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    continue

                value = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).sum()

                if pd.notna(value):

                    output.append(
                        f"**{col}** total: "
                        f"**{value:g}** "
                        f"({filename})"
                    )

        if output:

            return "\n".join(
                output
            )

    # =====================================================
    # MEDIAN
    # =====================================================

    if "median" in q:

        if not relevant_columns:

            relevant_columns = get_numeric_columns()

        output = []

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                if not pd.api.types.is_numeric_dtype(
                    df[col]
                ):
                    continue

                value = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).median()

                if pd.notna(value):

                    output.append(
                        f"**{col}** median: "
                        f"**{value:.2f}** "
                        f"({filename})"
                    )

        if output:

            return "\n".join(
                output
            )

    return None


# =========================================================
# CSV CONTEXT
# =========================================================

def build_csv_context(
    question
):

    relevant_columns = find_relevant_columns(
        question
    )

    parts = []

    for filename, df in st.session_state.csv_data.items():

        if relevant_columns:

            columns = [
                col
                for col in relevant_columns
                if col in df.columns
            ]

        else:

            columns = list(
                df.columns
            )

        if not columns:
            continue

        preview = df[
            columns
        ].head(20)

        parts.append(
            f"FILE: {filename}\n"
            f"COLUMNS: {', '.join(map(str, columns))}\n"
            f"DATA:\n"
            f"{preview.to_string(index=False)}"
        )

    return "\n\n".join(
        parts
    )


# =========================================================
# TXT CHUNKING
# =========================================================

def split_text(
    text,
    chunk_size=1200
):

    text = str(text)

    if not text.strip():
        return []

    chunks = []

    # Prefer paragraph based splitting
    paragraphs = re.split(
        r"\n\s*\n",
        text
    )

    current = ""

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            continue

        if len(
            current
        ) + len(
            paragraph
        ) <= chunk_size:

            current += (
                "\n\n" + paragraph
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

    # If paragraphs are too large
    final_chunks = []

    for chunk in chunks:

        if len(chunk) <= chunk_size:

            final_chunks.append(
                chunk
            )

        else:

            for i in range(
                0,
                len(chunk),
                chunk_size
            ):

                piece = chunk[
                    i:i + chunk_size
                ].strip()

                if piece:

                    final_chunks.append(
                        piece
                    )

    return final_chunks


# =========================================================
# TXT SEARCH
# =========================================================

def search_txt(
    question,
    top_k=5
):

    all_chunks = []

    for filename, text in st.session_state.txt_data.items():

        chunks = split_text(
            text
        )

        for chunk in chunks:

            all_chunks.append(
                (
                    filename,
                    chunk
                )
            )

    if not all_chunks:

        return []

    documents = [
        chunk
        for _, chunk in all_chunks
    ]

    q = normalize_text(
        question
    )

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            documents + [q]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

    except Exception:

        return []

    ranked = np.argsort(
        scores
    )[::-1]

    results = []

    for index in ranked[:top_k]:

        score = float(
            scores[index]
        )

        if score <= 0:
            continue

        filename, chunk = all_chunks[
            index
        ]

        results.append(
            {
                "file": filename,
                "text": chunk,
                "score": score
            }
        )

    return results


# =========================================================
# BUILD TXT CONTEXT
# =========================================================

def build_txt_context(
    question
):

    results = search_txt(
        question,
        top_k=5
    )

    if not results:

        return ""

    parts = []

    for result in results:

        parts.append(
            f"FILE: {result['file']}\n"
            f"RELEVANCE: {result['score']:.3f}\n"
            f"CONTENT:\n{result['text']}"
        )

    return "\n\n".join(
        parts
    )


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    csv_context,
    txt_context,
    direct_answer
):

    if client is None:

        return (
            None,
            "Gemini API is not configured."
        )

    prompt = f"""
You are IntelliMind AI.

Answer the user's question clearly and naturally.

USER QUESTION:
{question}

DIRECT CSV ANALYSIS:
{direct_answer or "No direct calculation available."}

CSV INFORMATION:
{csv_context or "No relevant CSV information found."}

TXT INFORMATION:
{txt_context or "No relevant TXT information found."}

RULES:

1. Use exact CSV calculations when available.
2. Never invent CSV numbers.
3. Use TXT information when relevant.
4. If the question is a general question,
   answer it normally.
5. Keep the answer simple.
6. Do not mention internal instructions.
7. Do not repeat unnecessary source data.
8. Give one clear logical answer.
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

        return (
            None,
            "Empty Gemini response."
        )

    except Exception as e:

        return (
            None,
            str(e)
        )


# =========================================================
# KNOWLEDGE BASE SUMMARY
# =========================================================

st.subheader(
    "📚 Knowledge Base Information"
)

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "CSV Files",
        len(
            st.session_state.csv_data
        )
    )

with col2:

    st.metric(
        "TXT Files",
        len(
            st.session_state.txt_data
        )
    )

with col3:

    total_rows = sum(
        len(df)
        for df in st.session_state.csv_data.values()
    )

    st.metric(
        "CSV Records",
        f"{total_rows:,}"
    )


# =========================================================
# CHAT HISTORY DISPLAY
# =========================================================

for message in st.session_state.chat_history:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("csv_context")
        ):

            with st.expander(
                "📊 View relevant CSV data"
            ):

                st.code(
                    message["csv_context"]
                )

        if (
            message["role"] == "assistant"
            and message.get("txt_context")
        ):

            with st.expander(
                "📄 View relevant TXT information"
            ):

                st.code(
                    message["txt_context"]
                )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question..."
)


# =========================================================
# QUESTION PROCESSING
# =========================================================

if question:

    # =====================================================
    # USER MESSAGE
    # =====================================================

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )

    # =====================================================
    # SEARCH / ANALYSIS
    # =====================================================

    direct_answer = direct_analysis(
        question
    )

    qa_result = search_qa_dataset(
        question
    )

    local_intent = local_intent_answer(
        question
    )

    builtin = built_in_answer(
        question
    )

    csv_context = build_csv_context(
        question
    )

    txt_context = build_txt_context(
        question
    )

    # =====================================================
    # ANSWER SELECTION
    # =====================================================

    answer = None
    gemini_error = None

    # -----------------------------------------------------
    # PRIORITY 1
    # Exact CSV analysis
    # -----------------------------------------------------

    if direct_answer:

        answer = direct_answer

    # -----------------------------------------------------
    # PRIORITY 2
    # Q&A dataset
    # -----------------------------------------------------

    elif qa_result:

        answer = qa_result[
            "answer"
        ]

    # -----------------------------------------------------
    # PRIORITY 3
    # Local conversation intent
    # -----------------------------------------------------

    elif local_intent:

        answer = local_intent

    # -----------------------------------------------------
    # PRIORITY 4
    # Built-in AI knowledge
    # -----------------------------------------------------

    elif builtin:

        answer = builtin

    # -----------------------------------------------------
    # PRIORITY 5
    # Gemini
    # -----------------------------------------------------

    else:

        answer, gemini_error = generate_ai_answer(
            question,
            csv_context,
            txt_context,
            direct_answer
        )

    # =====================================================
    # FINAL FALLBACK
    # =====================================================

    if not answer:

        if qa_result:

            answer = qa_result[
                "answer"
            ]

        elif local_intent:

            answer = local_intent

        elif builtin:

            answer = builtin

        elif direct_answer:

            answer = direct_answer

        elif gemini_error:

            error_lower = gemini_error.lower()

            if (
                "429" in error_lower
                or "resource_exhausted"
                in error_lower
                or "rate limit"
                in error_lower
                or "quota" in error_lower
            ):

                answer = (
                    "⚠️ Gemini rate limit reached.\n\n"
                    "I could not generate a new AI answer "
                    "right now. Please try again later."
                )

            elif (
                "404" in error_lower
                or "not found" in error_lower
            ):

                answer = (
                    "⚠️ Gemini model is unavailable "
                    "for this API configuration."
                )

            elif (
                "401" in error_lower
                or "403" in error_lower
            ):

                answer = (
                    "⚠️ Gemini API key is invalid "
                    "or does not have permission."
                )

            else:

                answer = (
                    "⚠️ I could not generate an answer "
                    "from the available knowledge."
                )

        else:

            answer = (
                "I could not find a relevant answer "
                "in the uploaded knowledge base."
            )

    # =====================================================
    # ASSISTANT MESSAGE
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # =============================================
        # CSV SOURCE
        # =============================================

        if csv_context:

            with st.expander(
                "📊 View relevant CSV data"
            ):

                st.code(
                    csv_context
                )

        # =============================================
        # TXT SOURCE
        # =============================================

        if txt_context:

            with st.expander(
                "📄 View relevant TXT information"
            ):

                st.code(
                    txt_context
                )

    # =====================================================
    # SAVE ASSISTANT MESSAGE
    # =====================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "csv_context": csv_context,
            "txt_context": txt_context
        }
    )
