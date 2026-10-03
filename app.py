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

        # Streamlit secrets
        try:
            api_key = st.secrets.get("GEMINI_API_KEY")
        except Exception:
            pass

        # Environment variable
        if not api_key:
            api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            return None

        return genai.Client(api_key=api_key)

    except Exception:
        return None


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text


def similarity(text1, text2):

    return difflib.SequenceMatcher(
        None,
        normalize_text(text1),
        normalize_text(text2)
    ).ratio()


def fuzzy_match(word, choices, cutoff=0.65):

    word = normalize_text(word)

    if not choices:
        return None

    best = None
    best_score = 0

    for choice in choices:

        score = similarity(word, choice)

        if score > best_score:
            best_score = score
            best = choice

    if best_score >= cutoff:
        return best

    return None


# =========================================================
# CSV FILE READING
# =========================================================

def read_csv_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file)

        return df

    except Exception:

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

            return df

        except Exception:

            try:

                uploaded_file.seek(0)

                df = pd.read_csv(
                    uploaded_file,
                    encoding="cp1252"
                )

                return df

            except Exception as e:

                st.error(f"CSV read error: {e}")

                return None


# =========================================================
# TXT FILE READING
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

    st.caption("CSV + TXT Intelligent Question Answering")

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
                f"**{filename}**  \n"
                f"{df.shape[0]} rows × {df.shape[1]} columns"
            )

    if st.session_state.txt_data:

        st.subheader("📄 TXT Files")

        for filename, text in st.session_state.txt_data.items():

            st.write(
                f"**{filename}**  \n"
                f"{len(text):,} characters"
            )

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# PROCESS UPLOADED FILES
# =========================================================

if uploaded_files:

    for uploaded_file in uploaded_files:

        filename = uploaded_file.name

        if filename.lower().endswith(".csv"):

            df = read_csv_file(uploaded_file)

            if df is not None:

                st.session_state.csv_data[filename] = df

        elif filename.lower().endswith(".txt"):

            text = read_txt_file(uploaded_file)

            st.session_state.txt_data[filename] = text


# =========================================================
# PAGE HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.write(
    "Ask questions from your uploaded CSV and TXT files."
)

st.caption(
    "CSV analysis • TXT knowledge base • Fuzzy matching • AI fallback"
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

if st.session_state.csv_data:

    st.subheader("📊 Dataset Overview")

    overview_data = []

    for filename, df in st.session_state.csv_data.items():

        overview_data.append({
            "File": filename,
            "Rows": df.shape[0],
            "Columns": df.shape[1],
            "Numeric Columns": len(df.select_dtypes(
                include=np.number
            ).columns),
            "Missing Values": int(df.isna().sum().sum())
        })

    overview_df = pd.DataFrame(overview_data)

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# FILE DETAILS
# =========================================================

if st.session_state.csv_data:

    with st.expander("📁 View CSV Files, Rows & Columns"):

        for filename, df in st.session_state.csv_data.items():

            st.markdown(f"### 📄 {filename}")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Rows", df.shape[0])

            with col2:
                st.metric("Columns", df.shape[1])

            with col3:
                st.metric(
                    "Missing Values",
                    int(df.isna().sum().sum())
                )

            st.write("**Column Names:**")

            st.write(
                ", ".join(
                    [str(c) for c in df.columns]
                )
            )

            st.write("**First 10 Rows:**")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            st.divider()


# =========================================================
# TXT DATASET OVERVIEW
# =========================================================

if st.session_state.txt_data:

    with st.expander("📄 View TXT Knowledge Base"):

        for filename, text in st.session_state.txt_data.items():

            st.markdown(f"### {filename}")

            st.write(
                f"Characters: **{len(text):,}**"
            )

            lines = text.splitlines()

            st.write(
                f"Lines: **{len(lines):,}**"
            )


# =========================================================
# TXT Q&A SEARCH
# =========================================================

def search_qa_dataset(question):

    best_answer = None
    best_question = None
    best_score = 0
    best_file = None

    question_normalized = normalize_text(question)

    for filename, text in st.session_state.txt_data.items():

        for line in text.splitlines():

            if "|" not in line:
                continue

            parts = line.split("|", 1)

            if len(parts) != 2:
                continue

            stored_question = parts[0].strip()
            stored_answer = parts[1].strip()

            if not stored_question or not stored_answer:
                continue

            stored_normalized = normalize_text(
                stored_question
            )

            # Exact match
            if question_normalized == stored_normalized:

                return {
                    "answer": stored_answer,
                    "matched_question": stored_question,
                    "score": 1.0,
                    "file": filename
                }

            # Similarity
            sim_score = similarity(
                question,
                stored_question
            )

            # Word overlap
            q_words = set(question_normalized.split())
            s_words = set(stored_normalized.split())

            overlap = 0

            if q_words:

                overlap = len(
                    q_words.intersection(s_words)
                ) / len(q_words)

            final_score = (
                sim_score * 0.7
                + overlap * 0.3
            )

            if final_score > best_score:

                best_score = final_score

                best_answer = stored_answer
                best_question = stored_question
                best_file = filename

    if best_score >= 0.68:

        return {
            "answer": best_answer,
            "matched_question": best_question,
            "score": best_score,
            "file": best_file
        }

    return None


# =========================================================
# LOCAL INTENT ANSWERS
# =========================================================

def local_intent_answer(question):

    q = normalize_text(question)

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

        if q == item or similarity(q, item) >= 0.88:

            return "Hello! 👋 How can I help you today?"

    if (
        similarity(q, "how are you") >= 0.78
        or similarity(q, "how are u") >= 0.78
    ):

        return (
            "I'm doing great! 😊 "
            "Thanks for asking. How can I help you?"
        )

    capability_questions = [
        "what can you do",
        "what do you do",
        "what are you doing",
        "what are you do",
        "what can u do"
    ]

    for item in capability_questions:

        if similarity(q, item) >= 0.75:

            return (
                "I can answer questions from your uploaded "
                "CSV and TXT files, perform basic data analysis, "
                "and answer general AI/ML/NLP questions."
            )

    if similarity(q, "can you help me") >= 0.78:

        return (
            "Of course! 😊 Ask me a question and "
            "I'll try to help."
        )

    chatbot_questions = [
        "are you a chatbot",
        "are you ai",
        "who are you"
    ]

    for item in chatbot_questions:

        if similarity(q, item) >= 0.75:

            return (
                "Yes! 🤖 I'm IntelliMind AI, "
                "an AI-powered question-answering chatbot."
            )

    thanks = [
        "thanks",
        "thank you",
        "thank u"
    ]

    for item in thanks:

        if similarity(q, item) >= 0.82:

            return "You're welcome! 😊"

    goodbye = [
        "goodbye",
        "bye",
        "see you"
    ]

    for item in goodbye:

        if similarity(q, item) >= 0.80:

            return "Goodbye! 👋 Have a great day!"

    return None


# =========================================================
# BUILT-IN AI / ML / DL / CV / NLP ANSWERS
# =========================================================

def built_in_answer(question):

    q = normalize_text(question)

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
                "AI is a field of computer science that allows "
                "computers to perform tasks that normally require "
                "human intelligence."
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
                "Machine Learning is a branch of AI that allows "
                "computers to learn patterns from data and make "
                "predictions or decisions."
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
                "Deep Learning is a subfield of Machine Learning "
                "that uses multi-layer neural networks to learn "
                "complex patterns from data."
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
                "Computer Vision is a field of AI that enables "
                "computers to understand and analyze images and videos."
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

            score = similarity(q, pattern)

            if q == normalize_text(pattern):

                return data["answer"]

            if score >= 0.80:

                return data["answer"]

    # Short-form direct questions
    short_forms = {
        "ai": "AI stands for Artificial Intelligence.",
        "ml": "ML stands for Machine Learning.",
        "dl": "DL stands for Deep Learning.",
        "cv": "CV stands for Computer Vision.",
        "nlp": "NLP stands for Natural Language Processing."
    }

    if q in short_forms:

        return short_forms[q]

    return None


# =========================================================
# CSV COLUMN HELPERS
# =========================================================

def get_all_columns():

    columns = []

    for filename, df in st.session_state.csv_data.items():

        for col in df.columns:

            if str(col) not in columns:

                columns.append(str(col))

    return columns


def get_numeric_columns():

    numeric_columns = []

    for filename, df in st.session_state.csv_data.items():

        for col in df.select_dtypes(
            include=np.number
        ).columns:

            if str(col) not in numeric_columns:

                numeric_columns.append(str(col))

    return numeric_columns


# =========================================================
# COLUMN SYNONYMS
# =========================================================

COLUMN_SYNONYMS = {

    "glucose": [
        "glucose",
        "blood sugar",
        "blood glucose",
        "sugar",
        "blood sugar level",
        "glucose level"
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
# FIND RELEVANT COLUMNS
# =========================================================

def find_relevant_columns(question):

    q = normalize_text(question)

    all_columns = get_all_columns()

    if not all_columns:
        return []

    found = []

    # Direct column matching
    for col in all_columns:

        col_normalized = normalize_text(col)

        if col_normalized in q:

            found.append(col)

    # Synonym matching
    for actual_column, words in COLUMN_SYNONYMS.items():

        matched_column = None

        for real_col in all_columns:

            if normalize_text(real_col) == actual_column:

                matched_column = real_col

                break

        if matched_column is None:
            continue

        for word in words:

            word_normalized = normalize_text(word)

            if word_normalized in q:

                if matched_column not in found:

                    found.append(matched_column)

                break

            if similarity(q, word_normalized) >= 0.80:

                if matched_column not in found:

                    found.append(matched_column)

                break

    return found


# =========================================================
# FIND BEST CSV
# =========================================================

def find_best_csv(question):

    if not st.session_state.csv_data:

        return None, None

    relevant_columns = find_relevant_columns(question)

    # If a specific column is mentioned
    if relevant_columns:

        for filename, df in st.session_state.csv_data.items():

            for col in relevant_columns:

                if col in df.columns:

                    return filename, df

    # Otherwise use first CSV
    filename = list(
        st.session_state.csv_data.keys()
    )[0]

    return (
        filename,
        st.session_state.csv_data[filename]
    )


# =========================================================
# NUMBER EXTRACTION
# =========================================================

def extract_number(question):

    numbers = re.findall(
        r"\b\d+(?:\.\d+)?\b",
        question
    )

    if numbers:

        try:
            return float(numbers[0])
        except Exception:
            pass

    return None


# =========================================================
# CSV DIRECT ANALYSIS
# =========================================================

def direct_csv_analysis(question):

    if not st.session_state.csv_data:

        return None

    q = normalize_text(question)

    filename, df = find_best_csv(question)

    if df is None:

        return None

    relevant_columns = find_relevant_columns(question)

    # =====================================================
    # ROW COUNT
    # =====================================================

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "rows are there" in q
        or "how many records" in q
        or "total records" in q
        or "how many patients" in q
        or "total patients" in q
    ):

        return {
            "answer": (
                f"The dataset **{filename}** contains "
                f"**{len(df):,} rows/records**."
            ),
            "filename": filename,
            "df": df,
            "columns": [],
            "analysis_type": "rows"
        }

    # =====================================================
    # COLUMN COUNT
    # =====================================================

    if (
        "how many columns" in q
        or "number of columns" in q
        or "total columns" in q
        or "how many fields" in q
    ):

        return {
            "answer": (
                f"The dataset **{filename}** has "
                f"**{len(df.columns)} columns**."
            ),
            "filename": filename,
            "df": df,
            "columns": list(df.columns),
            "analysis_type": "columns"
        }

    # =====================================================
    # COLUMN NAMES
    # =====================================================

    if (
        "column names" in q
        or "what are the columns" in q
        or "list columns" in q
        or "show columns" in q
        or "columns are" in q
    ):

        columns = ", ".join(
            [str(c) for c in df.columns]
        )

        return {
            "answer": (
                f"The columns in **{filename}** are:\n\n"
                f"{columns}"
            ),
            "filename": filename,
            "df": df,
            "columns": list(df.columns),
            "analysis_type": "columns"
        }

    # =====================================================
    # DATASET SHAPE
    # =====================================================

    if (
        "shape" in q
        or "dataset size" in q
        or "size of dataset" in q
    ):

        return {
            "answer": (
                f"The dataset **{filename}** has "
                f"**{df.shape[0]:,} rows** and "
                f"**{df.shape[1]} columns**."
            ),
            "filename": filename,
            "df": df,
            "columns": list(df.columns),
            "analysis_type": "shape"
        }

    # =====================================================
    # MISSING VALUES
    # =====================================================

    if (
        "missing" in q
        or "null" in q
        or "empty values" in q
        or "missing values" in q
    ):

        missing = int(
            df.isna().sum().sum()
        )

        missing_columns = (
            df.isna().sum()
            .sort_values(ascending=False)
        )

        missing_columns = (
            missing_columns[
                missing_columns > 0
            ]
        )

        if len(missing_columns) > 0:

            details = "\n".join(
                [
                    f"- {col}: {int(count)}"
                    for col, count
                    in missing_columns.items()
                ]
            )

            answer = (
                f"**{filename}** has "
                f"**{missing:,} missing values** in total.\n\n"
                f"Column-wise missing values:\n{details}"
            )

        else:

            answer = (
                f"**{filename}** has no missing values."
            )

        return {
            "answer": answer,
            "filename": filename,
            "df": df,
            "columns": [],
            "analysis_type": "missing"
        }

    # =====================================================
    # UNIQUE VALUES
    # =====================================================

    if "unique values" in q:

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                count = df[col].nunique(
                    dropna=True
                )

                return {
                    "answer": (
                        f"Column **{col}** has "
                        f"**{count:,} unique values**."
                    ),
                    "filename": filename,
                    "df": df,
                    "columns": [col],
                    "analysis_type": "unique"
                }

        return None

    # =====================================================
    # DIABETIC PATIENT COUNT
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

            numeric_outcome = pd.to_numeric(
                df[outcome_col],
                errors="coerce"
            )

            diabetic_count = int(
                (numeric_outcome == 1).sum()
            )

            return {
                "answer": (
                    f"According to **{filename}**, "
                    f"there are **{diabetic_count:,} "
                    f"diabetic patients** "
                    f"(Outcome = 1)."
                ),
                "filename": filename,
                "df": df,
                "columns": [outcome_col],
                "analysis_type": "diabetic_count"
            }

    # =====================================================
    # AVERAGE
    # =====================================================

    if (
        "average" in q
        or "avg" in q
        or "mean" in q
        or "avrage" in q
        or "averge" in q
        or "mean value" in q
    ):

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    avg = values.mean()

                    return {
                        "answer": (
                            f"The average **{col}** "
                            f"is **{avg:.2f}**."
                        ),
                        "filename": filename,
                        "df": df,
                        "columns": [col],
                        "analysis_type": "average"
                    }

        # If no specific column found
        numeric_cols = list(
            df.select_dtypes(
                include=np.number
            ).columns
        )

        if len(numeric_cols) == 1:

            col = numeric_cols[0]

            avg = df[col].mean()

            return {
                "answer": (
                    f"The average **{col}** "
                    f"is **{avg:.2f}**."
                ),
                "filename": filename,
                "df": df,
                "columns": [col],
                "analysis_type": "average"
            }

    # =====================================================
    # MAXIMUM
    # =====================================================

    if (
        "highest" in q
        or "higest" in q
        or "maximum" in q
        or "max" in q
        or "largest" in q
        or "greatest" in q
        or "highest value" in q
    ):

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    maximum = values.max()

                    return {
                        "answer": (
                            f"The highest **{col}** "
                            f"value is **{maximum}**."
                        ),
                        "filename": filename,
                        "df": df,
                        "columns": [col],
                        "analysis_type": "maximum"
                    }

    # =====================================================
    # MINIMUM
    # =====================================================

    if (
        "lowest" in q
        or "lowset" in q
        or "minimum" in q
        or "min" in q
        or "smallest" in q
        or "lowest value" in q
    ):

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    minimum = values.min()

                    return {
                        "answer": (
                            f"The lowest **{col}** "
                            f"value is **{minimum}**."
                        ),
                        "filename": filename,
                        "df": df,
                        "columns": [col],
                        "analysis_type": "minimum"
                    }

    # =====================================================
    # SUM
    # =====================================================

    if (
        "sum of" in q
        or "total of" in q
        or "total value" in q
        or "sum" in q
    ):

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    total = values.sum()

                    return {
                        "answer": (
                            f"The total **{col}** "
                            f"is **{total:,.2f}**."
                        ),
                        "filename": filename,
                        "df": df,
                        "columns": [col],
                        "analysis_type": "sum"
                    }

    # =====================================================
    # MEDIAN
    # =====================================================

    if "median" in q:

        if relevant_columns:

            col = relevant_columns[0]

            if col in df.columns:

                values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                ).dropna()

                if len(values) > 0:

                    median = values.median()

                    return {
                        "answer": (
                            f"The median **{col}** "
                            f"is **{median:.2f}**."
                        ),
                        "filename": filename,
                        "df": df,
                        "columns": [col],
                        "analysis_type": "median"
                    }

    return None


# =========================================================
# CSV FILTER / ROW SEARCH
# =========================================================

def search_csv_rows(question):

    if not st.session_state.csv_data:

        return None

    q = normalize_text(question)

    for filename, df in st.session_state.csv_data.items():

        # Example:
        # show patient age 50
        # glucose 120
        number = extract_number(question)

        relevant_columns = find_relevant_columns(question)

        if number is not None and relevant_columns:

            for col in relevant_columns:

                if col not in df.columns:
                    continue

                numeric_values = pd.to_numeric(
                    df[col],
                    errors="coerce"
                )

                matches = df[
                    numeric_values == number
                ]

                if len(matches) > 0:

                    return {
                        "filename": filename,
                        "df": df,
                        "rows": matches.head(10),
                        "columns": [col],
                        "answer": (
                            f"I found **{len(matches):,} matching "
                            f"row(s)** in `{filename}` where "
                            f"**{col} = {number:g}**."
                        )
                    }

    return None


# =========================================================
# CSV CONTEXT FOR GEMINI
# =========================================================

def build_csv_context(question):

    if not st.session_state.csv_data:

        return ""

    context_parts = []

    for filename, df in st.session_state.csv_data.items():

        relevant_columns = find_relevant_columns(
            question
        )

        if relevant_columns:

            available = [
                col
                for col in relevant_columns
                if col in df.columns
            ]

            if available:

                sample = df[available].head(15)

            else:

                sample = df.head(10)

        else:

            sample = df.head(10)

        context_parts.append(
            f"""
FILE: {filename}

ROWS: {len(df)}

COLUMNS: {list(df.columns)}

SAMPLE DATA:
{sample.to_string(index=False)}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# TXT CHUNKING
# =========================================================

def split_text(text, chunk_size=1200):

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

        if len(current) + len(paragraph) <= chunk_size:

            current += "\n" + paragraph

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
# TXT SEARCH
# =========================================================

def search_txt(question, top_k=5):

    if not st.session_state.txt_data:

        return []

    documents = []
    metadata = []

    for filename, text in st.session_state.txt_data.items():

        chunks = split_text(text)

        for chunk in chunks:

            documents.append(chunk)

            metadata.append(filename)

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
                    "score": float(scores[idx]),
                    "file": metadata[idx]
                })

        return results

    except Exception:

        return []


# =========================================================
# TXT CONTEXT
# =========================================================

def build_txt_context(question):

    results = search_txt(
        question,
        top_k=5
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

    return "\n".join(context)


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    csv_context,
    txt_context,
    direct_answer=None
):

    client = get_gemini_client()

    if client is None:

        return None, "API key not configured"

    prompt = f"""
You are IntelliMind AI, a helpful question-answering assistant.

User question:
{question}

IMPORTANT RULES:

1. If CSV data is provided, answer using the CSV data.
2. Never invent CSV numbers.
3. If the question asks for average, maximum, minimum, total,
   count, rows, columns, or other numerical information,
   use the provided data.
4. If TXT context is provided, use it when relevant.
5. If both CSV and TXT are irrelevant, answer as a normal
   educational assistant.
6. Keep the answer simple and clear.
7. If calculations are already provided by the application,
   use those values.
8. Do not say you cannot access the uploaded file.
9. If the user asks about AI, ML, DL, CV, NLP, explain simply.
10. Do not make up facts.

DIRECT CSV ANALYSIS:
{direct_answer if direct_answer else "None"}

CSV DATA:
{csv_context if csv_context else "No CSV data available."}

TXT KNOWLEDGE:
{txt_context if txt_context else "No TXT knowledge available."}

Answer the user's question now.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip(), None

        return None, "Empty AI response"

    except Exception as e:

        error_text = str(e)

        return None, error_text


# =========================================================
# DISPLAY CSV RESULT
# =========================================================

def display_csv_result(result):

    if not result:
        return

    df = result.get("df")

    filename = result.get(
        "filename",
        "CSV"
    )

    columns = result.get(
        "columns",
        []
    )

    st.success(result["answer"])

    if df is not None:

        st.caption(
            f"Source: {filename}"
        )

        if columns:

            valid_columns = [
                c
                for c in columns
                if c in df.columns
            ]

            if valid_columns:

                st.write(
                    "### 🔎 Relevant Column(s)"
                )

                st.dataframe(
                    df[valid_columns].head(10),
                    use_container_width=True
                )

        with st.expander(
            "📊 View Dataset Information"
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"**Rows:** {len(df):,}"
                )

            with col2:

                st.write(
                    f"**Columns:** {len(df.columns)}"
                )

            st.write(
                "**Columns:**"
            )

            st.write(
                ", ".join(
                    map(str, df.columns)
                )
            )

            st.write(
                "**Sample Rows:**"
            )

            st.dataframe(
                df.head(10),
                use_container_width=True
            )


# =========================================================
# CHAT DISPLAY
# =========================================================

for chat in st.session_state.chat_history:

    with st.chat_message(
        chat["role"]
    ):

        st.markdown(
            chat["content"]
        )

        if chat.get("csv_result"):

            result = chat["csv_result"]

            with st.expander(
                "📊 CSV Evidence"
            ):

                df = result.get("df")

                if df is not None:

                    columns = result.get(
                        "columns",
                        []
                    )

                    if columns:

                        valid_columns = [
                            c
                            for c in columns
                            if c in df.columns
                        ]

                        if valid_columns:

                            st.dataframe(
                                df[
                                    valid_columns
                                ].head(10),
                                use_container_width=True
                            )

                    st.caption(
                        f"File: {result.get('filename', '')}"
                    )

        if chat.get("txt_result"):

            with st.expander(
                "📄 TXT Source"
            ):

                st.write(
                    chat["txt_result"]
                )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your CSV or TXT file..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # User message
    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)

    # =====================================================
    # PRIORITY 1: DIRECT CSV ANALYSIS
    # =====================================================

    csv_result = direct_csv_analysis(
        question
    )

    if csv_result:

        answer = csv_result["answer"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "csv_result": csv_result
        })

        with st.chat_message("assistant"):

            display_csv_result(
                csv_result
            )

        st.stop()

    # =====================================================
    # PRIORITY 2: CSV ROW SEARCH
    # =====================================================

    row_result = search_csv_rows(
        question
    )

    if row_result:

        answer = row_result["answer"]

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "csv_result": row_result
        })

        with st.chat_message("assistant"):

            st.success(answer)

            st.write(
                "### 🔎 Matching Rows"
            )

            st.dataframe(
                row_result["rows"],
                use_container_width=True
            )

        st.stop()

    # =====================================================
    # PRIORITY 3: TXT Q&A
    # =====================================================

    qa_result = search_qa_dataset(
        question
    )

    if qa_result:

        answer = qa_result["answer"]

        display_answer = (
            f"{answer}\n\n"
            f"📄 **Source:** `{qa_result['file']}`"
        )

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": display_answer,
            "txt_result": (
                f"Matched question: "
                f"{qa_result['matched_question']}\n\n"
                f"Similarity: "
                f"{qa_result['score']:.2f}"
            )
        })

        with st.chat_message("assistant"):

            st.success(answer)

            st.caption(
                f"📄 Source: {qa_result['file']} "
                f"| Match: {qa_result['score']:.2f}"
            )

        st.stop()

    # =====================================================
    # PRIORITY 4: LOCAL INTENT
    # =====================================================

    local_answer = local_intent_answer(
        question
    )

    if local_answer:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": local_answer
        })

        with st.chat_message("assistant"):

            st.markdown(local_answer)

        st.stop()

    # =====================================================
    # PRIORITY 5: BUILT-IN KNOWLEDGE
    # =====================================================

    builtin = built_in_answer(
        question
    )

    if builtin:

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": builtin
        })

        with st.chat_message("assistant"):

            st.markdown(builtin)

        st.stop()

    # =====================================================
    # PRIORITY 6: GEMINI
    # =====================================================

    csv_context = build_csv_context(
        question
    )

    txt_context = build_txt_context(
        question
    )

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Thinking..."
        ):

            ai_answer, error = generate_ai_answer(
                question,
                csv_context,
                txt_context
            )

        if ai_answer:

            st.markdown(ai_answer)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": ai_answer
            })

        else:

            # =================================================
            # LOCAL FALLBACK
            # =================================================

            fallback = (
                "I couldn't generate an AI response right now. "
                "But you can ask me about your uploaded CSV/TXT "
                "data using questions like:\n\n"
                "- How many rows are there?\n"
                "- How many columns are there?\n"
                "- What are the column names?\n"
                "- What is the average glucose?\n"
                "- What is the highest glucose?\n"
                "- How many diabetic patients are there?\n"
                "- What is ML?\n"
                "- What is AI?\n"
            )

            st.warning(
                "⚠️ AI service is temporarily unavailable."
            )

            st.markdown(fallback)

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": fallback
            })

            if error:

                with st.expander(
                    "Technical information"
                ):

                    st.code(
                        error
                    )
