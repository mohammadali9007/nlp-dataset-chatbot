import os
import re
import difflib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from pypdf import PdfReader
from docx import Document

try:
    from google import genai
except ImportError:
    genai = None


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_NAME = "gemini-3.8-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "files_data" not in st.session_state:
    st.session_state.files_data = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "smart_questions" not in st.session_state:
    st.session_state.smart_questions = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .info-card {
        padding: 1rem;
        border-radius: 14px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(128,128,128,0.05);
    }

    .answer-card {
        padding: 1.2rem;
        border-radius: 16px;
        border: 1px solid rgba(128,128,128,0.2);
        margin-top: 0.5rem;
    }

    .field-info {
        font-size: 0.85rem;
        color: #6b7280;
        margin-top: 0.4rem;
    }

    .source-info {
        font-size: 0.78rem;
        color: #8b8b8b;
        margin-top: 0.6rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        padding: 12px;
        border-radius: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# COLUMN INFORMATION
# =========================================================

COLUMN_INFO = {

    # Diabetes
    "pregnancies": {
        "name": "Pregnancies",
        "meaning": "Number of pregnancies"
    },
    "glucose": {
        "name": "Glucose",
        "meaning": "Blood glucose concentration"
    },
    "bloodpressure": {
        "name": "Blood Pressure",
        "meaning": "Diastolic blood pressure"
    },
    "skinthickness": {
        "name": "Skin Thickness",
        "meaning": "Triceps skin fold thickness"
    },
    "insulin": {
        "name": "Insulin",
        "meaning": "Serum insulin level"
    },
    "bmi": {
        "name": "Body Mass Index",
        "meaning": "Body mass index"
    },
    "diabetespedigreefunction": {
        "name": "Diabetes Pedigree Function",
        "meaning": "Diabetes hereditary risk score"
    },
    "age": {
        "name": "Age",
        "meaning": "Age of the person"
    },
    "outcome": {
        "name": "Outcome",
        "meaning": "Diabetes outcome/class"
    },

    # Kidney
    "id": {
        "name": "ID",
        "meaning": "Record identifier"
    },
    "bp": {
        "name": "Blood Pressure",
        "meaning": "Blood pressure"
    },
    "sg": {
        "name": "Specific Gravity",
        "meaning": "Urine specific gravity"
    },
    "al": {
        "name": "Albumin",
        "meaning": "Albumin level"
    },
    "su": {
        "name": "Sugar",
        "meaning": "Urine sugar level"
    },
    "rbc": {
        "name": "Red Blood Cells",
        "meaning": "Red blood cell condition"
    },
    "pc": {
        "name": "Pus Cell",
        "meaning": "Pus cell condition"
    },
    "pcc": {
        "name": "Pus Cell Clumps",
        "meaning": "Presence of pus cell clumps"
    },
    "ba": {
        "name": "Bacteria",
        "meaning": "Presence of bacteria"
    },
    "bgr": {
        "name": "Blood Glucose Random",
        "meaning": "Random blood glucose"
    },
    "bu": {
        "name": "Blood Urea",
        "meaning": "Blood urea level"
    },
    "sc": {
        "name": "Serum Creatinine",
        "meaning": "Serum creatinine level"
    },
    "sod": {
        "name": "Sodium",
        "meaning": "Blood sodium level"
    },
    "pot": {
        "name": "Potassium",
        "meaning": "Blood potassium level"
    },
    "hemo": {
        "name": "Hemoglobin",
        "meaning": "Hemoglobin level"
    },
    "pcv": {
        "name": "Packed Cell Volume",
        "meaning": "Packed cell volume"
    },
    "wc": {
        "name": "White Blood Cell Count",
        "meaning": "White blood cell count"
    },
    "rc": {
        "name": "Red Blood Cell Count",
        "meaning": "Red blood cell count"
    },
    "htn": {
        "name": "Hypertension",
        "meaning": "Hypertension status"
    },
    "dm": {
        "name": "Diabetes Mellitus",
        "meaning": "Diabetes mellitus status"
    },
    "cad": {
        "name": "Coronary Artery Disease",
        "meaning": "Coronary artery disease status"
    },
    "appet": {
        "name": "Appetite",
        "meaning": "Appetite condition"
    },
    "pe": {
        "name": "Pedal Edema",
        "meaning": "Pedal edema status"
    },
    "ane": {
        "name": "Anemia",
        "meaning": "Anemia status"
    },
    "classification": {
        "name": "Classification",
        "meaning": "Kidney disease classification"
    },

    # General
    "salary": {
        "name": "Salary",
        "meaning": "Salary or income value"
    },
    "income": {
        "name": "Income",
        "meaning": "Income value"
    },
    "gpa": {
        "name": "GPA",
        "meaning": "Grade Point Average"
    },
    "score": {
        "name": "Score",
        "meaning": "Score or numeric result"
    },
    "marks": {
        "name": "Marks",
        "meaning": "Academic marks"
    },
    "price": {
        "name": "Price",
        "meaning": "Price value"
    }
}


# =========================================================
# ALIASES
# =========================================================

COLUMN_ALIASES = {

    "bp": [
        "bp",
        "blood pressure",
        "bloodpressure",
        "blood-pressure",
        "pressure"
    ],

    "bgr": [
        "bgr",
        "blood glucose random",
        "random glucose",
        "random blood glucose"
    ],

    "bu": [
        "bu",
        "blood urea",
        "urea"
    ],

    "sc": [
        "sc",
        "serum creatinine",
        "creatinine"
    ],

    "hemo": [
        "hemo",
        "hemoglobin",
        "haemoglobin"
    ],

    "bmi": [
        "bmi",
        "body mass index"
    ],

    "age": [
        "age"
    ],

    "glucose": [
        "glucose",
        "blood glucose"
    ],

    "insulin": [
        "insulin"
    ],

    "salary": [
        "salary",
        "pay",
        "income"
    ],

    "income": [
        "income",
        "earnings"
    ],

    "gpa": [
        "gpa",
        "grade point average"
    ],

    "marks": [
        "marks",
        "mark",
        "score"
    ]
}


# =========================================================
# BASIC HELPERS
# =========================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower().strip()
    text = re.sub(r"[_\-]+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text


def column_key(column):
    return re.sub(
        r"[^a-z0-9]",
        "",
        normalize_text(column)
    )


def get_column_info(column):
    key = column_key(column)

    if key in COLUMN_INFO:
        return COLUMN_INFO[key]

    return {
        "name": str(column),
        "meaning": "Dataset field"
    }


def clean_value(value):
    if pd.isna(value):
        return None

    return value


# =========================================================
# READ FILES
# =========================================================

def read_csv_file(uploaded_file):

    try:
        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file)

        df.columns = [
            str(col).strip()
            for col in df.columns
        ]

        return df

    except Exception as e:

        try:
            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

            df.columns = [
                str(col).strip()
                for col in df.columns
            ]

            return df

        except Exception:
            return None


def read_txt_file(uploaded_file):

    try:
        uploaded_file.seek(0)

        return uploaded_file.read().decode(
            "utf-8",
            errors="ignore"
        )

    except Exception:
        return ""


def read_pdf_file(uploaded_file):

    text = ""

    try:
        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        for page in reader.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    except Exception:
        pass

    return text


def read_docx_file(uploaded_file):

    text = ""

    try:
        uploaded_file.seek(0)

        document = Document(uploaded_file)

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                text += paragraph.text + "\n"

    except Exception:
        pass

    return text


# =========================================================
# PROCESS UPLOADED FILES
# =========================================================

def process_uploaded_files(uploaded_files):

    processed = []

    for file in uploaded_files:

        if file is None:
            continue

        name = getattr(file, "name", "Unknown file")

        extension = os.path.splitext(name)[1].lower()

        item = {
            "name": name,
            "type": extension,
            "df": None,
            "text": "",
            "size": getattr(file, "size", 0),
        }

        if extension == ".csv":

            df = read_csv_file(file)

            if df is not None:

                item["df"] = df

        elif extension == ".txt":

            item["text"] = read_txt_file(file)

        elif extension == ".pdf":

            item["text"] = read_pdf_file(file)

        elif extension == ".docx":

            item["text"] = read_docx_file(file)

        else:
            continue

        processed.append(item)

    return processed


# =========================================================
# TXT Q&A DATASET
# =========================================================

def parse_qa_dataset(text):

    pairs = []

    if not text:
        return pairs

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        if "|" not in line:
            continue

        question, answer = line.split("|", 1)

        question = question.strip()
        answer = answer.strip()

        if question and answer:

            pairs.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    return pairs


def search_qa_dataset(question, text):

    pairs = parse_qa_dataset(text)

    if not pairs:
        return None

    q = normalize_text(question)

    # Exact match
    for item in pairs:

        stored_q = normalize_text(
            item["question"]
        )

        if q == stored_q:

            return item["answer"]

    # Similarity matching
    questions = [
        normalize_text(item["question"])
        for item in pairs
    ]

    if not questions:
        return None

    try:

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            questions + [q]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        best_index = int(
            np.argmax(scores)
        )

        best_score = float(
            scores[best_index]
        )

        if best_score >= 0.45:

            return pairs[best_index]["answer"]

    except Exception:
        pass

    return None


# =========================================================
# DOCUMENT SEARCH
# =========================================================

def split_text(text, chunk_size=900):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    if not text:
        return []

    chunks = []

    words = text.split()

    current = []

    current_length = 0

    for word in words:

        current.append(word)

        current_length += len(word) + 1

        if current_length >= chunk_size:

            chunks.append(
                " ".join(current)
            )

            current = []
            current_length = 0

    if current:
        chunks.append(
            " ".join(current)
        )

    return chunks


def search_document(question, text):

    chunks = split_text(text)

    if not chunks:
        return None

    q = normalize_text(question)

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            chunks + [q]
        )

        scores = cosine_similarity(
            matrix[-1],
            matrix[:-1]
        )[0]

        best_index = int(
            np.argmax(scores)
        )

        best_score = float(
            scores[best_index]
        )

        if best_score >= 0.10:

            return chunks[best_index]

    except Exception:
        pass

    return None


# =========================================================
# IMPORTANT:
# QUESTION → EXACT COLUMN DETECTION
# =========================================================

def find_relevant_columns(question, df):

    if df is None or df.empty:
        return []

    q = normalize_text(question)

    q_no_space = q.replace(" ", "")

    q_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            q
        )
    )

    columns = list(df.columns)

    exact_matches = []

    # -----------------------------------------------------
    # STEP 1: EXACT / ALIAS MATCH
    # -----------------------------------------------------

    for col in columns:

        key = column_key(col)

        info = get_column_info(col)

        full_name = normalize_text(
            info["name"]
        )

        aliases = COLUMN_ALIASES.get(
            key,
            []
        )

        candidates = [
            key,
            normalize_text(col),
            full_name
        ]

        candidates.extend(
            normalize_text(alias)
            for alias in aliases
        )

        found = False

        for candidate in candidates:

            candidate_no_space = (
                candidate.replace(" ", "")
            )

            if not candidate_no_space:
                continue

            # Short field names:
            # bp, age, bmi, sc, etc.
            if len(candidate_no_space) <= 5:

                if candidate in q_words:
                    found = True
                    break

                if candidate_no_space in q_no_space:

                    # avoid false matching inside words
                    if (
                        len(candidate_no_space) <= 3
                        or re.search(
                            rf"\b{re.escape(candidate)}\b",
                            q
                        )
                    ):
                        found = True
                        break

            else:

                if candidate in q:
                    found = True
                    break

                if candidate_no_space in q_no_space:
                    found = True
                    break

        if found:

            exact_matches.append(col)

    if exact_matches:

        return exact_matches

    # -----------------------------------------------------
    # STEP 2: SAFE FUZZY MATCH
    #
    # IMPORTANT:
    # Do NOT fuzzy match tiny column names like
    # "age" against "average bp".
    # -----------------------------------------------------

    cleaned_question = q

    operation_words = [
        "average",
        "avg",
        "mean",
        "maximum",
        "max",
        "highest",
        "minimum",
        "min",
        "lowest",
        "median",
        "mode",
        "sum",
        "total",
        "count",
        "number",
        "how many",
        "show",
        "display",
        "give",
        "tell",
        "what is",
        "what are",
        "value",
        "values",
        "column",
        "field"
    ]

    for word in operation_words:

        cleaned_question = re.sub(
            rf"\b{re.escape(word)}\b",
            " ",
            cleaned_question
        )

    cleaned_question = re.sub(
        r"\s+",
        " ",
        cleaned_question
    ).strip()

    if not cleaned_question:
        return []

    matches = []

    for col in columns:

        key = column_key(col)

        # Never fuzzy match very short names
        if len(key) < 4:
            continue

        info = get_column_info(col)

        names = [
            normalize_text(col),
            normalize_text(info["name"])
        ]

        aliases = COLUMN_ALIASES.get(
            key,
            []
        )

        names.extend(
            normalize_text(x)
            for x in aliases
        )

        best_score = 0

        for name in names:

            if not name:
                continue

            score = difflib.SequenceMatcher(
                None,
                cleaned_question,
                name
            ).ratio()

            best_score = max(
                best_score,
                score
            )

        if best_score >= 0.80:

            matches.append(
                (
                    col,
                    best_score
                )
            )

    matches.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        item[0]
        for item in matches[:2]
    ]


# =========================================================
# QUESTION TYPE
# =========================================================

def detect_operation(question):

    q = normalize_text(question)

    if any(
        word in q
        for word in [
            "average",
            "avg",
            "mean"
        ]
    ):
        return "average"

    if any(
        word in q
        for word in [
            "maximum",
            "max",
            "highest",
            "largest"
        ]
    ):
        return "max"

    if any(
        word in q
        for word in [
            "minimum",
            "min",
            "lowest",
            "smallest"
        ]
    ):
        return "min"

    if "median" in q:
        return "median"

    if "mode" in q:
        return "mode"

    if any(
        word in q
        for word in [
            "sum",
            "total"
        ]
    ):
        return "sum"

    if any(
        word in q
        for word in [
            "unique",
            "distinct"
        ]
    ):
        return "unique"

    if any(
        word in q
        for word in [
            "missing",
            "null",
            "empty"
        ]
    ):
        return "missing"

    if any(
        word in q
        for word in [
            "type",
            "datatype",
            "data type"
        ]
    ):
        return "dtype"

    if any(
        word in q
        for word in [
            "first value",
            "first"
        ]
    ):
        return "first"

    if any(
        word in q
        for word in [
            "last value",
            "last"
        ]
    ):
        return "last"

    return None


# =========================================================
# DIRECT CSV ANALYSIS
# =========================================================

def analyze_csv_question(question, df):

    if df is None:
        return None, []

    if df.empty:
        return "The dataset is empty.", []

    q = normalize_text(question)

    # -----------------------------------------------------
    # GENERIC DATASET QUESTIONS
    # -----------------------------------------------------

    if any(
        phrase in q
        for phrase in [
            "how many rows",
            "number of rows",
            "total rows",
            "row count"
        ]
    ):

        return (
            f"The dataset has {len(df):,} rows.",
            []
        )

    if any(
        phrase in q
        for phrase in [
            "how many columns",
            "number of columns",
            "total columns",
            "column count"
        ]
    ):

        return (
            f"The dataset has {len(df.columns):,} columns.",
            []
        )

    if (
        "column names" in q
        or "columns are" in q
        or "what columns" in q
        or q == "columns"
    ):

        names = ", ".join(
            str(col)
            for col in df.columns
        )

        return (
            f"The columns are: {names}.",
            []
        )

    # -----------------------------------------------------
    # EXACT FIELD DETECTION
    # -----------------------------------------------------

    columns = find_relevant_columns(
        question,
        df
    )

    # VERY IMPORTANT:
    # If no column is confidently found,
    # DO NOT choose a random column.
    if not columns:
        return None, []

    column = columns[0]

    operation = detect_operation(question)

    series = df[column]

    numeric_series = pd.to_numeric(
        series,
        errors="coerce"
    )

    numeric_values = numeric_series.dropna()

    info = get_column_info(column)

    # -----------------------------------------------------
    # AVERAGE
    # -----------------------------------------------------

    if operation == "average":

        if numeric_values.empty:

            return (
                f"I couldn't calculate the average of "
                f"{info['name']} because it is not numeric.",
                [column]
            )

        value = numeric_values.mean()

        return (
            f"The average {info['name']} is "
            f"{value:.2f}.",
            [column]
        )

    # -----------------------------------------------------
    # MAX
    # -----------------------------------------------------

    if operation == "max":

        if numeric_values.empty:

            return (
                f"I couldn't calculate the maximum of "
                f"{info['name']} because it is not numeric.",
                [column]
            )

        value = numeric_values.max()

        return (
            f"The maximum {info['name']} is "
            f"{value:g}.",
            [column]
        )

    # -----------------------------------------------------
    # MIN
    # -----------------------------------------------------

    if operation == "min":

        if numeric_values.empty:

            return (
                f"I couldn't calculate the minimum of "
                f"{info['name']} because it is not numeric.",
                [column]
            )

        value = numeric_values.min()

        return (
            f"The minimum {info['name']} is "
            f"{value:g}.",
            [column]
        )

    # -----------------------------------------------------
    # MEDIAN
    # -----------------------------------------------------

    if operation == "median":

        if numeric_values.empty:

            return (
                f"I couldn't calculate the median of "
                f"{info['name']} because it is not numeric.",
                [column]
            )

        value = numeric_values.median()

        return (
            f"The median {info['name']} is "
            f"{value:.2f}.",
            [column]
        )

    # -----------------------------------------------------
    # MODE
    # -----------------------------------------------------

    if operation == "mode":

        modes = series.dropna().mode()

        if modes.empty:

            return (
                f"No mode is available for "
                f"{info['name']}.",
                [column]
            )

        values = [
            str(x)
            for x in modes.tolist()
        ]

        return (
            f"The mode of {info['name']} is "
            f"{', '.join(values)}.",
            [column]
        )

    # -----------------------------------------------------
    # SUM
    # -----------------------------------------------------

    if operation == "sum":

        if numeric_values.empty:

            return (
                f"I couldn't calculate the total of "
                f"{info['name']} because it is not numeric.",
                [column]
            )

        value = numeric_values.sum()

        return (
            f"The total {info['name']} is "
            f"{value:g}.",
            [column]
        )

    # -----------------------------------------------------
    # UNIQUE
    # -----------------------------------------------------

    if operation == "unique":

        count = series.nunique(
            dropna=True
        )

        return (
            f"{info['name']} has "
            f"{count:,} unique values.",
            [column]
        )

    # -----------------------------------------------------
    # MISSING
    # -----------------------------------------------------

    if operation == "missing":

        count = int(
            series.isna().sum()
        )

        return (
            f"{info['name']} has "
            f"{count:,} missing values.",
            [column]
        )

    # -----------------------------------------------------
    # DATA TYPE
    # -----------------------------------------------------

    if operation == "dtype":

        return (
            f"The data type of {info['name']} "
            f"is {series.dtype}.",
            [column]
        )

    # -----------------------------------------------------
    # FIRST
    # -----------------------------------------------------

    if operation == "first":

        values = series.dropna()

        if values.empty:

            return (
                f"There is no available value for "
                f"{info['name']}.",
                [column]
            )

        return (
            f"The first available {info['name']} "
            f"value is {values.iloc[0]}.",
            [column]
        )

    # -----------------------------------------------------
    # LAST
    # -----------------------------------------------------

    if operation == "last":

        values = series.dropna()

        if values.empty:

            return (
                f"There is no available value for "
                f"{info['name']}.",
                [column]
            )

        return (
            f"The last available {info['name']} "
            f"value is {values.iloc[-1]}.",
            [column]
        )

    # -----------------------------------------------------
    # "WHAT IS BP?"
    # -----------------------------------------------------

    if (
        q.startswith("what is")
        or q.startswith("what are")
        or q.startswith("tell me about")
        or q.startswith("explain")
    ):

        return (
            f"{info['name']} refers to "
            f"{info['meaning'].lower()}.",
            [column]
        )

    return None, []


# =========================================================
# GENERAL BUILT-IN ANSWERS
# =========================================================

GENERAL_ANSWERS = {

    "what is python":
        "Python is a high-level programming language known for its simple syntax and wide use in AI, data science, web development, and automation.",

    "what is ai":
        "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that normally require human intelligence.",

    "what is artificial intelligence":
        "Artificial Intelligence (AI) is the field of computer science focused on building systems that can learn, reason, understand information, and perform intelligent tasks.",

    "what is ml":
        "Machine Learning (ML) is a branch of AI where computers learn patterns from data and use those patterns to make predictions or decisions.",

    "what is machine learning":
        "Machine Learning is a branch of AI that allows computers to learn patterns from data without being explicitly programmed for every task.",

    "what is dl":
        "Deep Learning is a part of Machine Learning that uses multi-layer neural networks to learn complex patterns from large amounts of data.",

    "what is deep learning":
        "Deep Learning is a Machine Learning technique that uses neural networks with multiple layers to learn complex patterns.",

    "what is nlp":
        "Natural Language Processing (NLP) is a field of AI that helps computers understand, process, and generate human language.",

    "what is natural language processing":
        "Natural Language Processing (NLP) is a field of AI that enables computers to work with human language, such as text and speech.",

    "what is computer vision":
        "Computer Vision is a field of AI that enables computers to understand and analyze images and videos.",

    "what is csv":
        "CSV stands for Comma-Separated Values. It is a simple file format commonly used to store tabular data.",

    "what is pandas":
        "Pandas is a Python library used for data manipulation and analysis. Its main structures are DataFrame and Series.",

    "what is streamlit":
        "Streamlit is a Python framework that makes it easy to build interactive data and AI web applications.",

    "what is chatbot":
        "A chatbot is a software application that communicates with users through text or voice and provides automated responses.",

    "what is rag":
        "RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from a knowledge source and uses it to generate an answer.",

    "what is llm":
        "LLM stands for Large Language Model. It is an AI model trained on large amounts of text to understand and generate human-like language."
}


def get_general_answer(question):

    q = normalize_text(question)

    if q in GENERAL_ANSWERS:
        return GENERAL_ANSWERS[q]

    # small typo tolerance
    keys = list(GENERAL_ANSWERS.keys())

    matches = difflib.get_close_matches(
        q,
        keys,
        n=1,
        cutoff=0.82
    )

    if matches:

        return GENERAL_ANSWERS[
            matches[0]
        ]

    return None


# =========================================================
# GEMINI
# =========================================================

def get_gemini_client():

    if genai is None:
        return None

    api_key = None

    try:
        api_key = st.secrets.get(
            "GEMINI_API_KEY"
        )
    except Exception:
        pass

    if not api_key:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:
        return None

    try:

        return genai.Client(
            api_key=api_key
        )

    except Exception:
        return None


def generate_ai_answer(
    question,
    context=""
):

    client = get_gemini_client()

    if client is None:
        return None

    prompt = f"""
You are IntelliMind AI, a professional multi-file data assistant.

Answer the user's question clearly and accurately.

IMPORTANT:
- Use the supplied context when it is relevant.
- Do not invent dataset values.
- If the context does not contain the answer, say that clearly.
- Keep the answer concise.
- Do not show internal reasoning.
- Do not mention retrieval, TF-IDF, cosine similarity, chunks, embeddings,
  or internal processing.

User question:
{question}

Available context:
{context}
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return response.text.strip()

    except Exception:
        return None

    return None


# =========================================================
# BUILD FILE CONTEXT
# =========================================================

def build_file_context(files_data):

    context_parts = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        name = item.get(
            "name",
            "Unknown"
        )

        df = item.get("df")

        text = item.get(
            "text",
            ""
        )

        if df is not None:

            context_parts.append(
                f"""
FILE: {name}

Columns:
{", ".join(map(str, df.columns))}

Rows:
{len(df)}
"""
            )

        elif text:

            preview = text[:5000]

            context_parts.append(
                f"""
FILE: {name}

CONTENT:
{preview}
"""
            )

    return "\n".join(
        context_parts
    )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(
    question,
    files_data
):

    question = question.strip()

    if not question:
        return {
            "answer": "Please enter a question.",
            "fields": [],
            "sources": []
        }

    # -----------------------------------------------------
    # 1. GENERAL BUILT-IN KNOWLEDGE
    # -----------------------------------------------------

    general_answer = get_general_answer(
        question
    )

    if general_answer:

        return {
            "answer": general_answer,
            "fields": [],
            "sources": []
        }

    # -----------------------------------------------------
    # 2. DIRECT CSV ANALYSIS
    # -----------------------------------------------------

    csv_answers = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get("df")

        if df is None:
            continue

        answer, fields = analyze_csv_question(
            question,
            df
        )

        if answer:

            csv_answers.append(
                {
                    "answer": answer,
                    "fields": fields,
                    "source": item.get(
                        "name",
                        "CSV"
                    )
                }
            )

    # If a field-specific CSV answer exists,
    # return ONLY relevant datasets.
    if csv_answers:

        answers = []

        all_fields = []
        sources = []

        for item in csv_answers:

            answers.append(
                item["answer"]
            )

            all_fields.extend(
                item["fields"]
            )

            sources.append(
                item["source"]
            )

        # Remove duplicate fields
        all_fields = list(
            dict.fromkeys(all_fields)
        )

        # Remove duplicate sources
        sources = list(
            dict.fromkeys(sources)
        )

        return {
            "answer": "\n\n".join(answers),
            "fields": all_fields,
            "sources": sources
        }

    # -----------------------------------------------------
    # 3. TXT Q&A
    # -----------------------------------------------------

    for item in files_data:

        if not isinstance(item, dict):
            continue

        text = item.get(
            "text",
            ""
        )

        file_type = item.get(
            "type",
            ""
        )

        if (
            file_type == ".txt"
            and "|" in text
        ):

            answer = search_qa_dataset(
                question,
                text
            )

            if answer:

                return {
                    "answer": answer,
                    "fields": [],
                    "sources": [
                        item.get(
                            "name",
                            "TXT"
                        )
                    ]
                }

    # -----------------------------------------------------
    # 4. DOCUMENT SEARCH
    # -----------------------------------------------------

    document_answers = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        text = item.get(
            "text",
            ""
        )

        file_type = item.get(
            "type",
            ""
        )

        if (
            file_type in [
                ".pdf",
                ".docx"
            ]
            and text
        ):

            result = search_document(
                question,
                text
            )

            if result:

                document_answers.append(
                    {
                        "source": item.get(
                            "name",
                            "Document"
                        ),
                        "text": result
                    }
                )

    if document_answers:

        context = "\n\n".join(
            f"Source: {x['source']}\n{x['text']}"
            for x in document_answers
        )

        ai_answer = generate_ai_answer(
            question,
            context
        )

        if ai_answer:

            return {
                "answer": ai_answer,
                "fields": [],
                "sources": [
                    x["source"]
                    for x in document_answers
                ]
            }

        # Gemini unavailable:
        return {
            "answer": document_answers[0]["text"],
            "fields": [],
            "sources": [
                x["source"]
                for x in document_answers
            ]
        }

    # -----------------------------------------------------
    # 5. GENERAL GEMINI ANSWER
    # -----------------------------------------------------

    context = build_file_context(
        files_data
    )

    ai_answer = generate_ai_answer(
        question,
        context
    )

    if ai_answer:

        return {
            "answer": ai_answer,
            "fields": [],
            "sources": []
        }

    # -----------------------------------------------------
    # 6. FINAL FALLBACK
    # -----------------------------------------------------

    return {
        "answer": (
            "I couldn't find a reliable answer in the "
            "uploaded files. Try asking about a specific "
            "field, value, document topic, or dataset."
        ),
        "fields": [],
        "sources": []
    }


# =========================================================
# DYNAMIC SMART QUESTIONS
# =========================================================

def generate_smart_questions(files_data):

    questions = []

    # -----------------------------------------------------
    # CSV BASED QUESTIONS
    # -----------------------------------------------------

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get("df")

        if df is None or df.empty:
            continue

        columns = list(df.columns)

        # numeric fields
        numeric_columns = []

        for col in columns:

            numeric = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if numeric.notna().sum() > 0:

                numeric_columns.append(
                    col
                )

        # create average suggestions
        for col in numeric_columns[:5]:

            info = get_column_info(col)

            questions.append(
                f"What is the average {info['name']}?"
            )

        # max / min
        for col in numeric_columns[:2]:

            info = get_column_info(col)

            questions.append(
                f"What is the maximum {info['name']}?"
            )

            questions.append(
                f"What is the minimum {info['name']}?"
            )

        # rows
        questions.append(
            f"How many rows are in {item.get('name', 'this dataset')}?"
        )

        questions.append(
            f"What columns are in {item.get('name', 'this dataset')}?"
        )

        # categorical columns
        categorical = []

        for col in columns:

            if col not in numeric_columns:

                categorical.append(
                    col
                )

        for col in categorical[:2]:

            info = get_column_info(col)

            questions.append(
                f"How many unique {info['name']} values are there?"
            )

    # -----------------------------------------------------
    # DOCUMENT QUESTIONS
    # -----------------------------------------------------

    for item in files_data:

        if not isinstance(item, dict):
            continue

        file_type = item.get(
            "type",
            ""
        )

        if file_type in [
            ".txt",
            ".pdf",
            ".docx"
        ]:

            name = item.get(
                "name",
                "this file"
            )

            questions.append(
                f"Summarize {name}"
            )

            questions.append(
                f"What information is available in {name}?"
            )

    # Remove duplicates
    unique_questions = list(
        dict.fromkeys(
            questions
        )
    )

    return unique_questions[:12]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "## 🤖 IntelliMind AI"
    )

    st.caption(
        "Your Multi-File Data Copilot"
    )

    st.divider()

    st.markdown(
        "### 📂 Upload Files"
    )

    uploaded_files = st.file_uploader(
        "CSV, TXT, PDF or DOCX",
        type=[
            "csv",
            "txt",
            "pdf",
            "docx"
        ],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:

        names = [
            file.name
            for file in uploaded_files
        ]

        old_names = st.session_state.get(
            "last_uploaded_names",
            []
        )

        if names != old_names:

            st.session_state.files_data = (
                process_uploaded_files(
                    uploaded_files
                )
            )

            st.session_state.last_uploaded_names = (
                names
            )

            st.session_state.chat_history = []

            st.session_state.smart_questions = (
                generate_smart_questions(
                    st.session_state.files_data
                )
            )

    files_data = st.session_state.files_data

    st.divider()

    # -----------------------------------------------------
    # FILE LIST
    # -----------------------------------------------------

    st.markdown(
        "### 📁 Files"
    )

    if not files_data:

        st.caption(
            "No files uploaded yet."
        )

    else:

        for index, item in enumerate(
            files_data
        ):

            if not isinstance(item, dict):
                continue

            file_name = item.get(
                "name",
                "Unknown file"
            )

            file_type = item.get(
                "type",
                ""
            )

            df = item.get(
                "df"
            )

            with st.expander(
                f"📄 {file_name}",
                expanded=False
            ):

                if df is not None:

                    st.caption(
                        f"{len(df):,} rows • "
                        f"{len(df.columns)} columns"
                    )

                    st.markdown(
                        "**All Columns**"
                    )

                    for col in df.columns:

                        info = get_column_info(
                            col
                        )

                        st.markdown(
                            f"**`{col}`**"
                        )

                        st.caption(
                            f"Full name: {info['name']}"
                        )

                        with st.expander(
                            "Meaning",
                            expanded=False
                        ):

                            st.caption(
                                info["meaning"]
                            )

                else:

                    text = item.get(
                        "text",
                        ""
                    )

                    st.caption(
                        f"{len(text):,} characters"
                    )

    st.divider()

    # -----------------------------------------------------
    # FIELD DICTIONARY
    # -----------------------------------------------------

    st.markdown(
        "### 📚 Fields"
    )

    displayed_fields = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get("df")

        if df is not None:

            for col in df.columns:

                key = column_key(col)

                if key in displayed_fields:
                    continue

                displayed_fields.append(
                    key
                )

                info = get_column_info(
                    col
                )

                st.markdown(
                    f"`{col}` → {info['name']}"
                )

                if len(displayed_fields) >= 8:
                    break

    if len(displayed_fields) > 0:

        with st.expander(
            "View all fields"
        ):

            already = set()

            for item in files_data:

                if not isinstance(item, dict):
                    continue

                df = item.get("df")

                if df is None:
                    continue

                for col in df.columns:

                    key = column_key(col)

                    if key in already:
                        continue

                    already.add(key)

                    info = get_column_info(
                        col
                    )

                    st.markdown(
                        f"`{col}` → {info['name']}"
                    )

                    st.caption(
                        info["meaning"]
                    )


# =========================================================
# MAIN HEADER
# =========================================================

st.markdown(
    '<div class="main-title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Your intelligent multi-file data and document assistant'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# EMPTY STATE
# =========================================================

if not files_data:

    st.info(
        "👈 Upload CSV, TXT, PDF or DOCX files from the sidebar to get started."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown(
            "### 📊 Analyze Data"
        )

        st.caption(
            "Ask about averages, maximums, minimums, rows, columns and more."
        )

    with col2:

        st.markdown(
            "### 📄 Understand Files"
        )

        st.caption(
            "Search and summarize TXT, PDF and DOCX documents."
        )

    with col3:

        st.markdown(
            "### 🧠 Ask Naturally"
        )

        st.caption(
            "Ask questions using normal language."
        )


# =========================================================
# DATASET OVERVIEW
# =========================================================

else:

    total_files = len(
        files_data
    )

    csv_files = sum(
        1
        for item in files_data
        if isinstance(item, dict)
        and item.get("df") is not None
    )

    document_files = total_files - csv_files

    total_rows = sum(
        len(item["df"])
        for item in files_data
        if isinstance(item, dict)
        and item.get("df") is not None
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Files",
            total_files
        )

    with c2:
        st.metric(
            "Datasets",
            csv_files
        )

    with c3:
        st.metric(
            "Documents",
            document_files
        )

    with c4:
        st.metric(
            "Total Rows",
            f"{total_rows:,}"
        )

    st.divider()


# =========================================================
# SMART QUESTIONS
# =========================================================

if files_data:

    st.markdown(
        "### ✨ Smart Questions"
    )

    if not st.session_state.smart_questions:

        st.caption(
            "Upload a dataset to generate questions automatically."
        )

    else:

        cols = st.columns(3)

        for index, question in enumerate(
            st.session_state.smart_questions
        ):

            with cols[index % 3]:

                if st.button(
                    question,
                    key=f"smart_{index}",
                    use_container_width=True
                ):

                    st.session_state.selected_question = (
                        question
                    )

                    st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

if st.session_state.chat_history:

    st.markdown(
        "### 💬 Conversation"
    )

    for message in st.session_state.chat_history:

        role = message.get(
            "role",
            "assistant"
        )

        content = message.get(
            "content",
            ""
        )

        if role == "user":

            with st.chat_message(
                "user"
            ):

                st.write(content)

        else:

            with st.chat_message(
                "assistant"
            ):

                st.write(content)

                fields = message.get(
                    "fields",
                    []
                )

                sources = message.get(
                    "sources",
                    []
                )

                if fields:

                    field_text = []

                    for field in fields:

                        info = get_column_info(
                            field
                        )

                        field_text.append(
                            f"`{field}` → {info['name']}"
                        )

                    st.caption(
                        " • ".join(
                            field_text
                        )
                    )

                if sources:

                    st.caption(
                        "Source: "
                        + ", ".join(
                            sources
                        )
                    )


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.chat_input(
    "Ask anything about your uploaded files..."
)

if st.session_state.selected_question:

    question = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = None


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    result = process_question(
        question,
        st.session_state.files_data
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "fields": result.get(
                "fields",
                []
            ),
            "sources": result.get(
                "sources",
                []
            )
        }
    )

    st.rerun()
