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
    st.session_state.files_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"


# =========================================================
# COLUMN DICTIONARY
# =========================================================

COLUMN_INFO = {

    "id": (
        "ID",
        "Unique identification number."
    ),

    "age": (
        "Age",
        "Age of a person."
    ),

    "bp": (
        "Blood Pressure",
        "Blood pressure measurement."
    ),

    "bloodpressure": (
        "Blood Pressure",
        "Blood pressure measurement."
    ),

    "bmi": (
        "Body Mass Index",
        "Measure calculated from height and weight."
    ),

    "glucose": (
        "Blood Glucose",
        "Amount of glucose in the blood."
    ),

    "pregnancies": (
        "Number of Pregnancies",
        "Number of pregnancies."
    ),

    "insulin": (
        "Insulin",
        "Hormone involved in blood glucose control."
    ),

    "outcome": (
        "Outcome",
        "Result or target classification."
    ),

    "classification": (
        "Classification",
        "Disease or target classification."
    ),

    "sc": (
        "Serum Creatinine",
        "Blood creatinine level."
    ),

    "bu": (
        "Blood Urea",
        "Amount of urea in the blood."
    ),

    "hemo": (
        "Hemoglobin",
        "Hemoglobin level."
    ),

    "pcv": (
        "Packed Cell Volume",
        "Percentage of blood occupied by red blood cells."
    ),

    "sod": (
        "Sodium",
        "Sodium level in blood."
    ),

    "pot": (
        "Potassium",
        "Potassium level in blood."
    ),

    "htn": (
        "Hypertension",
        "High blood pressure condition."
    ),

    "dm": (
        "Diabetes Mellitus",
        "Diabetes condition."
    ),

    "cad": (
        "Coronary Artery Disease",
        "Disease affecting coronary arteries."
    ),

    "rbc": (
        "Red Blood Cells",
        "Red blood cell information."
    ),

    "pc": (
        "Pus Cells",
        "Pus cell information."
    ),

    "pcc": (
        "Pus Cell Clumps",
        "Presence of pus cell clumps."
    ),

    "ba": (
        "Bacteria",
        "Presence of bacteria."
    ),

    "bgr": (
        "Random Blood Glucose",
        "Random blood glucose measurement."
    ),

    "sg": (
        "Specific Gravity",
        "Measure of urine concentration."
    ),

    "al": (
        "Albumin",
        "Albumin level."
    ),

    "su": (
        "Sugar",
        "Urine sugar measurement."
    ),

    "pe": (
        "Pedal Edema",
        "Swelling of feet or legs."
    ),

    "ane": (
        "Anemia",
        "Condition related to low hemoglobin or red blood cells."
    ),

    "appet": (
        "Appetite",
        "Appetite condition."
    ),

    "name": (
        "Name",
        "Name of a person or item."
    ),

    "gender": (
        "Gender",
        "Gender information."
    ),

    "email": (
        "Email Address",
        "Email address."
    ),

    "phone": (
        "Phone Number",
        "Contact phone number."
    ),

    "address": (
        "Address",
        "Residential or business address."
    ),

    "gpa": (
        "Grade Point Average",
        "Academic grade average."
    ),

    "cgpa": (
        "Cumulative Grade Point Average",
        "Overall academic grade average."
    ),

    "salary": (
        "Salary",
        "Regular payment received from employment."
    ),

    "income": (
        "Income",
        "Money received from work or other sources."
    ),

    "annualincome": (
        "Annual Income",
        "Total income received in one year."
    ),

    "experience": (
        "Work Experience",
        "Amount of professional experience."
    ),

    "department": (
        "Department",
        "Academic or organizational department."
    ),

    "score": (
        "Score",
        "Numerical score."
    ),

    "rating": (
        "Rating",
        "Rating value."
    ),

    "price": (
        "Price",
        "Cost or monetary value."
    ),

    "quantity": (
        "Quantity",
        "Number or amount of items."
    ),

    "height": (
        "Height",
        "Height measurement."
    ),

    "weight": (
        "Weight",
        "Weight measurement."
    ),

    "status": (
        "Status",
        "Current state or condition."
    ),

    "performance_score": (
        "Performance Score",
        "Numerical performance measurement."
    ),

    "credit_score": (
        "Credit Score",
        "Numerical measure of creditworthiness."
    ),

    "loan_status": (
        "Loan Status",
        "Current status of a loan."
    ),

    "ai": (
        "Artificial Intelligence",
        "Artificial Intelligence."
    ),

    "ml": (
        "Machine Learning",
        "Machine Learning."
    ),

    "dl": (
        "Deep Learning",
        "Deep Learning."
    ),

    "nlp": (
        "Natural Language Processing",
        "Natural Language Processing."
    ),

    "cv": (
        "Computer Vision",
        "Computer Vision."
    )
}


# =========================================================
# TEXT HELPERS
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


def clean_column_name(column):

    value = str(column).strip()

    value = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        value
    )

    value = value.replace(
        "_",
        " "
    )

    value = value.replace(
        "-",
        " "
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


def column_key(column):

    return re.sub(
        r"[^a-z0-9]",
        "",
        str(column).lower()
    )


def get_column_info(column):

    key = column_key(column)

    if key in COLUMN_INFO:
        return COLUMN_INFO[key]

    readable = clean_column_name(
        column
    ).title()

    return (
        readable,
        f"Information related to {readable}."
    )


def similarity(a, b):

    return difflib.SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


# =========================================================
# FILE READERS
# =========================================================

def read_csv_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        return pd.read_csv(
            uploaded_file
        )

    except Exception:

        try:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

        except Exception:

            try:

                uploaded_file.seek(0)

                return pd.read_csv(
                    uploaded_file,
                    encoding="cp1252"
                )

            except Exception:

                return None


def read_txt_file(uploaded_file):

    raw = uploaded_file.read()

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]:

        try:
            return raw.decode(encoding)

        except Exception:
            continue

    return raw.decode(
        "utf-8",
        errors="ignore"
    )


def read_pdf_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        reader = PdfReader(
            uploaded_file
        )

        pages = []

        for page in reader.pages:

            try:

                text = page.extract_text()

                if text:
                    pages.append(text)

            except Exception:
                pass

        return "\n".join(pages)

    except Exception:

        return ""


def read_docx_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        document = Document(
            uploaded_file
        )

        parts = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                parts.append(text)

        for table in document.tables:

            for row in table.rows:

                row_text = " | ".join(
                    cell.text.strip()
                    for cell in row.cells
                )

                if row_text.strip():
                    parts.append(row_text)

        return "\n".join(parts)

    except Exception:

        return ""


# =========================================================
# FILE TYPE ICON
# =========================================================

def get_file_icon(file_type):

    icons = {
        "csv": "📊",
        "txt": "📄",
        "pdf": "📕",
        "docx": "📝"
    }

    return icons.get(
        file_type,
        "📁"
    )


# =========================================================
# PARSE TXT QUESTION | ANSWER
# =========================================================

def parse_qa_text(text):

    pairs = []

    if not text:
        return pairs

    for line in text.splitlines():

        line = line.strip()

        if "|" not in line:
            continue

        question, answer = line.split(
            "|",
            1
        )

        question = question.strip()
        answer = answer.strip()

        if question and answer:

            pairs.append(
                (
                    question,
                    answer
                )
            )

    return pairs


def search_qa_dataset(
    question,
    text
):

    pairs = parse_qa_text(
        text
    )

    if not pairs:
        return None

    normalized_question = normalize_text(
        question
    )

    # Exact match
    for q, answer in pairs:

        if normalize_text(q) == normalized_question:

            return answer

    # Fuzzy match
    best_answer = None
    best_score = 0

    for q, answer in pairs:

        score = similarity(
            question,
            q
        )

        if score > best_score:

            best_score = score
            best_answer = answer

    if best_score >= 0.68:

        return best_answer

    return None


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(
    text,
    chunk_size=900
):

    if not text:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(
            r"\n\s*\n",
            text
        )
        if p.strip()
    ]

    if not paragraphs:

        paragraphs = [
            x.strip()
            for x in text.splitlines()
            if x.strip()
        ]

    chunks = []

    current = ""

    for paragraph in paragraphs:

        if len(current) + len(paragraph) <= chunk_size:

            current += " " + paragraph

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
# SEARCH ONE DOCUMENT
# =========================================================

def search_document(
    question,
    text,
    top_k=3
):

    chunks = split_text(
        text
    )

    if not chunks:
        return []

    if len(chunks) == 1:
        return chunks

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            chunks
        )

        q_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            q_vector,
            matrix
        )[0]

        indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in indices:

            if scores[idx] >= 0.08:

                results.append(
                    chunks[idx]
                )

        return results

    except Exception:

        return []


# =========================================================
# SEARCH ALL TEXT FILES
# =========================================================

def search_all_documents(
    question
):

    results = []

    for filename, data in st.session_state.files_data.items():

        file_type = data.get(
            "type"
        )

        if file_type == "csv":
            continue

        text = data.get(
            "text",
            ""
        )

        if not text:
            continue

        # First try Q&A format
        qa_answer = search_qa_dataset(
            question,
            text
        )

        if qa_answer:

            results.append({
                "file": filename,
                "type": "qa",
                "content": qa_answer,
                "score": 1.0
            })

            continue

        # Generic document search
        chunks = search_document(
            question,
            text
        )

        for chunk in chunks:

            score = similarity(
                question,
                chunk[:500]
            )

            results.append({
                "file": filename,
                "type": file_type,
                "content": chunk,
                "score": score
            })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:8]


# =========================================================
# FIND CSV COLUMN MATCHES
# =========================================================

def find_column_matches(
    question,
    df
):

    if df is None:
        return []

    q = normalize_text(
        question
    )

    matches = []

    for col in df.columns:

        original = str(col)

        readable, _ = get_column_info(
            original
        )

        names = [
            normalize_text(original),
            normalize_text(
                clean_column_name(original)
            ),
            normalize_text(readable)
        ]

        matched = False

        for name in names:

            if not name:
                continue

            if name in q:
                matched = True
                break

        if matched:

            matches.append(
                original
            )

    return list(
        dict.fromkeys(matches)
    )


# =========================================================
# DIRECT CSV ANALYSIS
# =========================================================

def direct_csv_analysis(
    question,
    df
):

    if df is None or df.empty:
        return None

    q = normalize_text(
        question
    )

    # -----------------------------------------------------
    # ROW COUNT
    # -----------------------------------------------------

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "how many records" in q
        or "total records" in q
    ):

        return (
            f"This dataset contains "
            f"**{len(df):,} rows**."
        )

    # -----------------------------------------------------
    # COLUMN COUNT
    # -----------------------------------------------------

    if (
        "how many columns" in q
        or "number of columns" in q
        or "total columns" in q
    ):

        return (
            f"This dataset contains "
            f"**{len(df.columns)} columns**."
        )

    # -----------------------------------------------------
    # SHAPE
    # -----------------------------------------------------

    if (
        "shape" in q
        or "rows and columns" in q
    ):

        return (
            f"The dataset shape is "
            f"**{df.shape[0]:,} rows × "
            f"{df.shape[1]} columns**."
        )

    # -----------------------------------------------------
    # COLUMN NAMES
    # -----------------------------------------------------

    if (
        "column names" in q
        or "what columns" in q
        or "list columns" in q
    ):

        names = ", ".join(
            str(x)
            for x in df.columns
        )

        return (
            f"The columns are:\n\n"
            f"**{names}**"
        )

    # -----------------------------------------------------
    # MISSING
    # -----------------------------------------------------

    if (
        "missing values" in q
        or "missing data" in q
        or "null values" in q
        or "empty values" in q
    ):

        missing = df.isna().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if missing.empty:

            return (
                "There are **no missing values** "
                "in this dataset."
            )

        lines = []

        for col, count in missing.items():

            lines.append(
                f"- **{col}:** {int(count):,}"
            )

        return (
            "### Missing Values\n\n"
            + "\n".join(lines)
        )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    if "duplicate" in q:

        duplicates = int(
            df.duplicated().sum()
        )

        return (
            f"This dataset contains "
            f"**{duplicates:,} duplicate rows**."
        )

    # -----------------------------------------------------
    # COLUMN MEANING
    # -----------------------------------------------------

    for col in df.columns:

        original = str(col)

        readable, meaning = get_column_info(
            original
        )

        col_normal = normalize_text(
            original
        )

        readable_normal = normalize_text(
            readable
        )

        if (
            col_normal in q
            or readable_normal in q
        ):

            if (
                "full form" in q
                or "meaning" in q
                or "what does" in q
                or "what is" in q
            ):

                return (
                    f"**{original}** → "
                    f"**{readable}**\n\n"
                    f"{meaning}"
                )

    # -----------------------------------------------------
    # NUMERIC ANALYSIS
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_columns:

        col_normal = normalize_text(
            col
        )

        readable, _ = get_column_info(
            col
        )

        readable_normal = normalize_text(
            readable
        )

        matched = (
            col_normal in q
            or readable_normal in q
        )

        if not matched:
            continue

        series = pd.to_numeric(
            df[col],
            errors="coerce"
        ).dropna()

        if series.empty:
            continue

        # Average
        if (
            "average" in q
            or "mean" in q
        ):

            return (
                f"The average **{clean_column_name(col)}** "
                f"is **{series.mean():,.2f}**."
            )

        # Highest
        if (
            "highest" in q
            or "maximum" in q
            or "max" in q
        ):

            return (
                f"The highest **{clean_column_name(col)}** "
                f"is **{series.max():,.2f}**."
            )

        # Lowest
        if (
            "lowest" in q
            or "minimum" in q
            or "min" in q
        ):

            return (
                f"The lowest **{clean_column_name(col)}** "
                f"is **{series.min():,.2f}**."
            )

        # Median
        if "median" in q:

            return (
                f"The median **{clean_column_name(col)}** "
                f"is **{series.median():,.2f}**."
            )

        # Total
        if (
            "total" in q
            or "sum" in q
        ):

            return (
                f"The total **{clean_column_name(col)}** "
                f"is **{series.sum():,.2f}**."
            )

    # -----------------------------------------------------
    # CATEGORICAL ANALYSIS
    # -----------------------------------------------------

    categorical_columns = [
        col
        for col in df.columns
        if col not in numeric_columns
    ]

    for col in categorical_columns:

        col_normal = normalize_text(
            col
        )

        readable, _ = get_column_info(
            col
        )

        readable_normal = normalize_text(
            readable
        )

        if (
            col_normal not in q
            and readable_normal not in q
        ):
            continue

        if (
            "common" in q
            or "most frequent" in q
            or "popular" in q
        ):

            counts = (
                df[col]
                .dropna()
                .astype(str)
                .value_counts()
                .head(5)
            )

            if counts.empty:
                return None

            lines = []

            for value, count in counts.items():

                lines.append(
                    f"- **{value}**: {int(count):,}"
                )

            return (
                f"### Most Common Values\n\n"
                + "\n".join(lines)
            )

        if "unique" in q:

            count = df[col].nunique(
                dropna=True
            )

            return (
                f"**{col}** contains "
                f"**{count:,} unique values**."
            )

    return None


# =========================================================
# MULTI CSV SEARCH
# =========================================================

def search_all_csv_files(
    question
):

    results = []

    for filename, data in st.session_state.files_data.items():

        if data.get("type") != "csv":
            continue

        df = data.get(
            "data"
        )

        if df is None:
            continue

        answer = direct_csv_analysis(
            question,
            df
        )

        if answer:

            matched_columns = find_column_matches(
                question,
                df
            )

            results.append({
                "file": filename,
                "answer": answer,
                "columns": matched_columns
            })

    return results


# =========================================================
# GENERAL DATASET CONTEXT
# =========================================================

def create_all_file_context(
    question
):

    context_parts = []

    # -----------------------------------------------------
    # ALL CSV FILES
    # -----------------------------------------------------

    for filename, data in st.session_state.files_data.items():

        if data.get("type") != "csv":
            continue

        df = data.get(
            "data"
        )

        if df is None:
            continue

        context_parts.append(
            f"""
===== CSV FILE: {filename} =====

Shape:
{df.shape}

Columns:
{list(df.columns)}

Data Types:
{df.dtypes.to_string()}

First 8 rows:
{df.head(8).to_string(index=False)}
"""
        )

    # -----------------------------------------------------
    # ALL DOCUMENT FILES
    # -----------------------------------------------------

    document_results = search_all_documents(
        question
    )

    for item in document_results:

        context_parts.append(
            f"""
===== DOCUMENT: {item['file']} =====

{item['content']}
"""
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# BUILT-IN ANSWERS
# =========================================================

BUILT_IN_ANSWERS = {

    "python": (
        "Python is a high-level programming language "
        "used for web development, automation, data science, "
        "machine learning and artificial intelligence."
    ),

    "machine learning": (
        "Machine Learning is a branch of AI where computers "
        "learn patterns from data and use those patterns "
        "to make predictions or decisions."
    ),

    "ml": (
        "ML stands for Machine Learning. "
        "It allows computers to learn patterns from data "
        "without being explicitly programmed for every task."
    ),

    "deep learning": (
        "Deep Learning is a type of Machine Learning "
        "that uses multi-layer neural networks."
    ),

    "dl": (
        "DL stands for Deep Learning. "
        "It uses neural networks with multiple layers."
    ),

    "nlp": (
        "NLP stands for Natural Language Processing. "
        "It enables computers to process and understand human language."
    ),

    "natural language processing": (
        "Natural Language Processing is an AI field "
        "focused on processing and understanding human language."
    ),

    "computer vision": (
        "Computer Vision is an AI field that enables "
        "computers to understand images and videos."
    ),

    "cv": (
        "CV can mean Computer Vision. "
        "It focuses on enabling computers to understand visual information."
    ),

    "ai": (
        "AI stands for Artificial Intelligence. "
        "It refers to computer systems that perform tasks "
        "normally requiring human intelligence."
    ),

    "artificial intelligence": (
        "Artificial Intelligence refers to computer systems "
        "that can perform tasks requiring human-like intelligence."
    )
}


def built_in_answer(
    question
):

    q = normalize_text(
        question
    )

    if q in BUILT_IN_ANSWERS:

        return BUILT_IN_ANSWERS[q]

    for key, answer in BUILT_IN_ANSWERS.items():

        if key in q:

            return answer

    return None


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    if genai is None:
        return None

    try:

        api_key = st.secrets.get(
            "GEMINI_API_KEY",
            ""
        )

        if not api_key:

            api_key = os.getenv(
                "GEMINI_API_KEY",
                ""
            )

        if not api_key:
            return None

        return genai.Client(
            api_key=api_key
        )

    except Exception:

        return None


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    context
):

    client = get_gemini_client()

    if client is None:
        return None

    language = (
        st.session_state.response_language
    )

    if language == "Bangla":

        language_rule = (
            "Answer in simple Bangla. "
            "Keep important technical terms in English."
        )

    elif language == "Auto":

        language_rule = (
            "Answer in the same language used by the user."
        )

    else:

        language_rule = (
            "Answer in simple English."
        )

    prompt = f"""
You are IntelliMind AI, a professional multi-file
Data Copilot and document assistant.

The user may upload multiple CSV, TXT, PDF and DOCX files.

IMPORTANT RULES:

1. Use information from ALL uploaded files when relevant.
2. You are allowed to combine information from multiple files.
3. If a question refers to multiple datasets, compare or combine them.
4. For numerical claims from CSV files, use only the supplied data.
5. Never invent numbers.
6. Do not hallucinate CSV values.
7. If a requested value cannot be calculated from the provided data, say so.
8. For documents, use the provided document context.
9. For general knowledge questions, answer normally.
10. Give one clear final answer instead of showing internal processing.
11. Do not show matching rows.
12. Do not show CSV evidence tables.
13. Do not explain the retrieval process.
14. Keep answers concise but useful.
15. {language_rule}

USER QUESTION:
{question}

UPLOADED FILE CONTEXT:
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
# DYNAMIC QUESTION GENERATOR
# =========================================================

def generate_multi_file_questions():

    questions = {
        "Overview": [],
        "Statistics": [],
        "Comparison": [],
        "Insights": [],
        "Data Quality": []
    }

    csv_files = [
        name
        for name, data
        in st.session_state.files_data.items()
        if data.get("type") == "csv"
    ]

    text_files = [
        name
        for name, data
        in st.session_state.files_data.items()
        if data.get("type") != "csv"
    ]

    # -----------------------------------------------------
    # GENERAL
    # -----------------------------------------------------

    questions["Overview"] = [
        "How many files have I uploaded?",
        "Give me an overview of all uploaded files.",
        "What information is available in these files?"
    ]

    # -----------------------------------------------------
    # CSV QUESTIONS
    # -----------------------------------------------------

    numeric_columns = []

    categorical_columns = []

    for filename in csv_files:

        df = st.session_state.files_data[
            filename
        ]["data"]

        for col in df.select_dtypes(
            include=np.number
        ).columns:

            numeric_columns.append(
                (filename, col)
            )

        for col in df.select_dtypes(
            exclude=np.number
        ).columns:

            categorical_columns.append(
                (filename, col)
            )

    # Statistics
    for filename, col in numeric_columns[:8]:

        questions["Statistics"].append(
            f"What is the average {col} in {filename}?"
        )

        questions["Statistics"].append(
            f"What is the highest {col} in {filename}?"
        )

    # Comparison
    if len(csv_files) >= 2:

        questions["Comparison"].extend([
            "Compare the uploaded datasets.",
            "What are the differences between these datasets?",
            "Which datasets have more rows?",
            "Compare the common columns across the datasets."
        ])

    if len(numeric_columns) >= 2:

        first_file, first_col = numeric_columns[0]
        second_file, second_col = numeric_columns[1]

        questions["Comparison"].append(
            f"Compare {first_col} from {first_file} "
            f"with {second_col} from {second_file}."
        )

    # Insights
    questions["Insights"].extend([
        "What are the main insights from all datasets?",
        "What interesting patterns can you find?"
    ])

    # Data quality
    questions["Data Quality"] = [
        "Are there missing values in the uploaded datasets?",
        "Which dataset has the most missing values?",
        "Are there duplicate rows?"
    ]

    # Document questions
    if text_files:

        questions["Overview"].extend([
            "Summarize the uploaded documents.",
            "What are the main topics in the documents?"
        ])

        questions["Insights"].extend([
            "What important information is present in the documents?"
        ])

    # Remove duplicates
    for category in questions:

        unique = []

        for q in questions[category]:

            if q not in unique:
                unique.append(q)

        questions[category] = unique[:10]

    return questions


# =========================================================
# MULTI FILE OVERVIEW
# =========================================================

def show_file_overview():

    total_files = len(
        st.session_state.files_data
    )

    csv_count = 0
    document_count = 0

    total_rows = 0

    for data in st.session_state.files_data.values():

        if data.get("type") == "csv":

            csv_count += 1

            df = data.get(
                "data"
            )

            if df is not None:
                total_rows += len(df)

        else:

            document_count += 1

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.metric(
            "Total Files",
            total_files
        )

    with c2:

        st.metric(
            "CSV Files",
            csv_count
        )

    with c3:

        st.metric(
            "Documents",
            document_count
        )

    with c4:

        st.metric(
            "Total CSV Rows",
            f"{total_rows:,}"
        )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 IntelliMind AI")

    st.caption(
        "Your Multi-File Data Copilot"
    )

    st.divider()

    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    st.subheader("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "CSV, TXT, PDF or DOCX",
        type=[
            "csv",
            "txt",
            "pdf",
            "docx"
        ],
        accept_multiple_files=True
    )

    if uploaded_files:

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            if filename in st.session_state.files_data:
                continue

            extension = (
                filename
                .lower()
                .split(".")[-1]
            )

            if extension == "csv":

                df = read_csv_file(
                    uploaded_file
                )

                if df is not None:

                    st.session_state.files_data[
                        filename
                    ] = {
                        "type": "csv",
                        "data": df
                    }

            elif extension == "txt":

                text = read_txt_file(
                    uploaded_file
                )

                st.session_state.files_data[
                    filename
                ] = {
                    "type": "txt",
                    "text": text
                }

            elif extension == "pdf":

                text = read_pdf_file(
                    uploaded_file
                )

                st.session_state.files_data[
                    filename
                ] = {
                    "type": "pdf",
                    "text": text
                }

            elif extension == "docx":

                text = read_docx_file(
                    uploaded_file
                )

                st.session_state.files_data[
                    filename
                ] = {
                    "type": "docx",
                    "text": text
                }

    # -----------------------------------------------------
    # UPLOADED FILES
    # -----------------------------------------------------

    if st.session_state.files_data:

        st.divider()

        st.subheader("📂 Uploaded Files")

        for filename, data in st.session_state.files_data.items():

            icon = get_file_icon(
                data.get("type")
            )

            st.caption(
                f"{icon} `{filename}`"
            )

    # -----------------------------------------------------
    # FIELD DICTIONARY
    # -----------------------------------------------------

    st.divider()

    st.subheader("📚 Fields")

    all_columns = []

    for filename, data in st.session_state.files_data.items():

        if data.get("type") == "csv":

            df = data.get(
                "data"
            )

            if df is not None:

                for col in df.columns:

                    if col not in all_columns:

                        all_columns.append(
                            col
                        )

    if all_columns:

        # Compact sidebar
        for col in all_columns[:8]:

            readable, _ = get_column_info(
                col
            )

            st.caption(
                f"`{col}` → {readable}"
            )

        if len(all_columns) > 8:

            st.caption(
                f"+ {len(all_columns) - 8} more fields"
            )

        with st.expander(
            "🔎 View all fields"
        ):

            for col in all_columns:

                readable, meaning = get_column_info(
                    col
                )

                st.markdown(
                    f"**`{col}`** → {readable}"
                )

                st.caption(
                    meaning
                )

    else:

        st.caption(
            "Upload CSV files to see fields."
        )

    # -----------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------

    st.divider()

    st.subheader("⚙️ Settings")

    st.session_state.response_language = st.selectbox(
        "Response Language",
        [
            "English",
            "Bangla",
            "Auto"
        ]
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()

    if st.button(
        "🧹 Remove All Files",
        use_container_width=True
    ):

        st.session_state.files_data = {}

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.caption(
    "Analyze multiple datasets and documents in one conversation."
)


# =========================================================
# NO FILE
# =========================================================

if not st.session_state.files_data:

    st.info(
        "👋 Upload multiple CSV, TXT, PDF or DOCX files from the sidebar."
    )

    st.markdown(
        """
### 🚀 What IntelliMind can do

- 📊 Analyze multiple CSV files
- 📄 Search multiple TXT files
- 📕 Understand PDF documents
- 📝 Read Word documents
- 🔢 Calculate real dataset statistics
- 🔍 Find information across multiple files
- 🔄 Compare different datasets
- 💡 Generate smart questions
- 🤖 Answer general AI/ML questions
"""
    )


# =========================================================
# FILES AVAILABLE
# =========================================================

else:

    show_file_overview()

    st.divider()

    # -----------------------------------------------------
    # QUICK FILE LIST
    # -----------------------------------------------------

    st.subheader("🗂️ Your Knowledge Workspace")

    file_columns = st.columns(
        min(
            4,
            len(st.session_state.files_data)
        )
    )

    for index, (
        filename,
        data
    ) in enumerate(
        st.session_state.files_data.items()
    ):

        with file_columns[
            index % len(file_columns)
        ]:

            icon = get_file_icon(
                data.get("type")
            )

            st.info(
                f"{icon} **{filename}**\n\n"
                f"`{data.get('type', '').upper()}`"
            )

    # -----------------------------------------------------
    # QUICK INSIGHTS
    # -----------------------------------------------------

    csv_files = [
        (
            filename,
            data
        )
        for filename, data
        in st.session_state.files_data.items()
        if data.get("type") == "csv"
    ]

    if csv_files:

        st.subheader("⚡ Quick Data Insights")

        total_rows = sum(
            len(data["data"])
            for filename, data
            in csv_files
            if data.get("data") is not None
        )

        total_columns = sum(
            len(data["data"].columns)
            for filename, data
            in csv_files
            if data.get("data") is not None
        )

        a, b, c = st.columns(3)

        with a:
            st.metric(
                "Datasets",
                len(csv_files)
            )

        with b:
            st.metric(
                "Combined Rows",
                f"{total_rows:,}"
            )

        with c:
            st.metric(
                "Total Fields",
                total_columns
            )

    # -----------------------------------------------------
    # SMART QUESTIONS
    # -----------------------------------------------------

    st.divider()

    st.subheader(
        "💡 Smart Questions"
    )

    questions = generate_multi_file_questions()

    tabs = st.tabs([
        "📊 Overview",
        "🔢 Statistics",
        "🔄 Comparison",
        "💡 Insights",
        "🛡️ Data Quality"
    ])

    categories = [
        "Overview",
        "Statistics",
        "Comparison",
        "Insights",
        "Data Quality"
    ]

    for tab, category in zip(
        tabs,
        categories
    ):

        with tab:

            category_questions = questions.get(
                category,
                []
            )

            if not category_questions:

                st.caption(
                    "No suggestions available."
                )

            else:

                for i in range(
                    0,
                    len(category_questions),
                    2
                ):

                    cols = st.columns(2)

                    for j in range(2):

                        index = i + j

                        if index >= len(
                            category_questions
                        ):
                            continue

                        q = category_questions[
                            index
                        ]

                        with cols[j]:

                            if st.button(
                                q,
                                key=f"{category}_{index}_{q}",
                                use_container_width=True
                            ):

                                st.session_state.selected_question = q

                                st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

st.divider()

st.subheader("💬 Conversation")

for item in st.session_state.chat_history:

    with st.chat_message(
        "user"
    ):

        st.write(
            item["question"]
        )

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            item["answer"]
        )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(
    question
):

    question = question.strip()

    if not question:
        return

    answer = None

    relevant_fields = []

    source_files = []

    # =====================================================
    # 1. SEARCH ALL CSV FILES
    # =====================================================

    csv_results = search_all_csv_files(
        question
    )

    if csv_results:

        source_files = [
            item["file"]
            for item in csv_results
        ]

        for item in csv_results:

            for col in item["columns"]:

                if col not in relevant_fields:

                    relevant_fields.append(
                        col
                    )

        # One CSV result
        if len(csv_results) == 1:

            answer = csv_results[0]["answer"]

        # Multiple CSV results
        else:

            parts = []

            for item in csv_results:

                parts.append(
                    f"**{item['file']}**\n\n"
                    f"{item['answer']}"
                )

            answer = (
                "### Results from multiple datasets\n\n"
                + "\n\n".join(parts)
            )

    # =====================================================
    # 2. SEARCH ALL TXT/PDF/DOCX
    # =====================================================

    document_results = search_all_documents(
        question
    )

    if document_results:

        # If no CSV direct answer
        if answer is None:

            best = document_results[0]

            answer = best["content"]

            source_files = [
                item["file"]
                for item in document_results[:4]
            ]

        # If both CSV + document information exists
        else:

            document_context = "\n\n".join(
                item["content"]
                for item in document_results[:4]
            )

            context = f"""
CSV RESULT:

{answer}

DOCUMENT INFORMATION:

{document_context}
"""

            # Let Gemini combine them
            ai_answer = generate_ai_answer(
                question,
                context
            )

            if ai_answer:

                answer = ai_answer

            source_files.extend(
                item["file"]
                for item in document_results[:4]
                if item["file"] not in source_files
            )

    # =====================================================
    # 3. BUILT-IN KNOWLEDGE
    # =====================================================

    if answer is None:

        answer = built_in_answer(
            question
        )

    # =====================================================
    # 4. GEMINI WITH ALL FILES
    # =====================================================

    if answer is None:

        all_context = create_all_file_context(
            question
        )

        answer = generate_ai_answer(
            question,
            all_context
        )

    # =====================================================
    # 5. FALLBACK
    # =====================================================

    if answer is None:

        answer = (
            "I couldn't find a reliable answer "
            "from the uploaded files or local knowledge. "
            "Please try asking the question differently."
        )

    # =====================================================
    # SAVE CHAT
    # =====================================================

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

    # =====================================================
    # DISPLAY USER
    # =====================================================

    with st.chat_message(
        "user"
    ):

        st.write(
            question
        )

    # =====================================================
    # DISPLAY ANSWER
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        # Only show relevant fields
        if relevant_fields:

            st.caption(
                "📌 Relevant fields"
            )

            for col in relevant_fields[:4]:

                readable, _ = get_column_info(
                    col
                )

                st.caption(
                    f"`{col}` → {readable}"
                )

        # Source files shown compactly
        if source_files:

            unique_sources = list(
                dict.fromkeys(
                    source_files
                )
            )

            st.caption(
                "📂 Source: "
                + ", ".join(
                    unique_sources[:5]
                )
            )


# =========================================================
# SELECTED SMART QUESTION
# =========================================================

if st.session_state.selected_question:

    question = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = ""

    process_question(
        question
    )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask anything about your uploaded files..."
)

if user_question:

    process_question(
        user_question
    )
