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
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_NAME = "gemini-3.8-flash"


# =========================================================
# SESSION STATE
# =========================================================

if "files_data" not in st.session_state:
    st.session_state.files_data = []

elif not isinstance(st.session_state.files_data, list):
    st.session_state.files_data = []


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

elif not isinstance(st.session_state.chat_history, list):
    st.session_state.chat_history = []


if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []


# =========================================================
# COLUMN DICTIONARY
# =========================================================

COLUMN_INFO = {

    # Diabetes
    "pregnancies": (
        "Pregnancies",
        "Number of pregnancies"
    ),

    "glucose": (
        "Glucose",
        "Plasma glucose concentration"
    ),

    "bloodpressure": (
        "Blood Pressure",
        "Diastolic blood pressure"
    ),

    "skinthickness": (
        "Skin Thickness",
        "Triceps skin fold thickness"
    ),

    "insulin": (
        "Insulin",
        "2-hour serum insulin"
    ),

    "bmi": (
        "Body Mass Index",
        "Body Mass Index"
    ),

    "diabetespedigreefunction": (
        "Diabetes Pedigree Function",
        "Diabetes hereditary risk score"
    ),

    "age": (
        "Age",
        "Age of the person"
    ),

    "outcome": (
        "Outcome",
        "Diabetes outcome or classification"
    ),

    # Kidney
    "id": (
        "ID",
        "Unique record identifier"
    ),

    "age": (
        "Age",
        "Age of the person"
    ),

    "bp": (
        "Blood Pressure",
        "Blood pressure"
    ),

    "sg": (
        "Specific Gravity",
        "Urine specific gravity"
    ),

    "al": (
        "Albumin",
        "Albumin level"
    ),

    "su": (
        "Sugar",
        "Urine sugar level"
    ),

    "rbc": (
        "Red Blood Cells",
        "Red blood cell condition"
    ),

    "pc": (
        "Pus Cell",
        "Pus cell condition"
    ),

    "pcc": (
        "Pus Cell Clumps",
        "Presence of pus cell clumps"
    ),

    "ba": (
        "Bacteria",
        "Presence of bacteria"
    ),

    "bgr": (
        "Blood Glucose Random",
        "Random blood glucose level"
    ),

    "bu": (
        "Blood Urea",
        "Blood urea level"
    ),

    "sc": (
        "Serum Creatinine",
        "Serum creatinine level"
    ),

    "sod": (
        "Sodium",
        "Serum sodium level"
    ),

    "pot": (
        "Potassium",
        "Serum potassium level"
    ),

    "hemo": (
        "Hemoglobin",
        "Hemoglobin level"
    ),

    "pcv": (
        "Packed Cell Volume",
        "Packed cell volume"
    ),

    "wc": (
        "White Blood Cell Count",
        "White blood cell count"
    ),

    "rc": (
        "Red Blood Cell Count",
        "Red blood cell count"
    ),

    "htn": (
        "Hypertension",
        "Whether hypertension is present"
    ),

    "dm": (
        "Diabetes Mellitus",
        "Whether diabetes mellitus is present"
    ),

    "cad": (
        "Coronary Artery Disease",
        "Whether coronary artery disease is present"
    ),

    "appet": (
        "Appetite",
        "Patient appetite"
    ),

    "pe": (
        "Pedal Edema",
        "Presence of pedal edema"
    ),

    "ane": (
        "Anemia",
        "Presence of anemia"
    ),

    "classification": (
        "Classification",
        "Final disease classification"
    ),

    # Common fields
    "name": (
        "Name",
        "Name of the person"
    ),

    "gender": (
        "Gender",
        "Gender of the person"
    ),

    "sex": (
        "Sex",
        "Sex of the person"
    ),

    "salary": (
        "Salary",
        "Salary amount"
    ),

    "income": (
        "Income",
        "Income amount"
    ),

    "department": (
        "Department",
        "Department name"
    ),

    "education": (
        "Education",
        "Education level"
    ),

    "experience": (
        "Experience",
        "Years of experience"
    ),

    "height": (
        "Height",
        "Height measurement"
    ),

    "weight": (
        "Weight",
        "Weight measurement"
    ),

    "city": (
        "City",
        "City name"
    ),

    "country": (
        "Country",
        "Country name"
    ),

    "email": (
        "Email",
        "Email address"
    ),

    "phone": (
        "Phone",
        "Phone number"
    ),

    "date": (
        "Date",
        "Date value"
    ),

    "price": (
        "Price",
        "Price amount"
    ),

    "quantity": (
        "Quantity",
        "Quantity value"
    ),

    "category": (
        "Category",
        "Category name"
    ),

    "rating": (
        "Rating",
        "Rating value"
    ),

    "score": (
        "Score",
        "Score value"
    ),

    "marks": (
        "Marks",
        "Marks obtained"
    ),

    "gpa": (
        "GPA",
        "Grade Point Average"
    )
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):

    text = str(text).lower()

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

    pretty = str(column)

    pretty = re.sub(
        r"[_\-]+",
        " ",
        pretty
    )

    pretty = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        pretty
    )

    pretty = pretty.strip()

    words = pretty.split()

    full_name = " ".join(
        word.capitalize()
        for word in words
    )

    return (
        full_name,
        "Dataset field"
    )


# =========================================================
# SIMILARITY
# =========================================================

def text_similarity(a, b):

    a = normalize_text(a)
    b = normalize_text(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    return difflib.SequenceMatcher(
        None,
        a,
        b
    ).ratio()


# =========================================================
# FILE READERS
# =========================================================

def read_csv_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file
        )

        return {
            "name": uploaded_file.name,
            "type": "csv",
            "data": df,
            "text": "",
            "columns": list(df.columns)
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"CSV reading error: {e}",
            "columns": []
        }


def read_txt_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        raw = uploaded_file.read()

        if isinstance(raw, bytes):

            raw = raw.decode(
                "utf-8",
                errors="ignore"
            )

        return {
            "name": uploaded_file.name,
            "type": "txt",
            "data": None,
            "text": raw,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"TXT reading error: {e}",
            "columns": []
        }


def read_pdf_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        reader = PdfReader(
            uploaded_file
        )

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        full_text = "\n".join(
            pages
        )

        return {
            "name": uploaded_file.name,
            "type": "pdf",
            "data": None,
            "text": full_text,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"PDF reading error: {e}",
            "columns": []
        }


def read_docx_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        document = Document(
            uploaded_file
        )

        paragraphs = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        full_text = "\n".join(
            paragraphs
        )

        return {
            "name": uploaded_file.name,
            "type": "docx",
            "data": None,
            "text": full_text,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"DOCX reading error: {e}",
            "columns": []
        }


# =========================================================
# PROCESS MULTIPLE FILES
# =========================================================

def process_uploaded_files(uploaded_files):

    results = []

    for uploaded_file in uploaded_files:

        try:

            filename = uploaded_file.name

            lower_name = filename.lower()

            if lower_name.endswith(".csv"):

                item = read_csv_file(
                    uploaded_file
                )

            elif lower_name.endswith(".txt"):

                item = read_txt_file(
                    uploaded_file
                )

            elif lower_name.endswith(".pdf"):

                item = read_pdf_file(
                    uploaded_file
                )

            elif lower_name.endswith(".docx"):

                item = read_docx_file(
                    uploaded_file
                )

            else:

                continue

            if not isinstance(
                item,
                dict
            ):
                continue

            # Ensure all required keys exist

            item.setdefault(
                "name",
                filename
            )

            item.setdefault(
                "type",
                "unknown"
            )

            item.setdefault(
                "data",
                None
            )

            item.setdefault(
                "text",
                ""
            )

            item.setdefault(
                "columns",
                []
            )

            results.append(
                item
            )

        except Exception as e:

            results.append(
                {
                    "name": uploaded_file.name,
                    "type": "error",
                    "data": None,
                    "text": str(e),
                    "columns": []
                }
            )

    return results


# =========================================================
# TXT QUESTION | ANSWER PARSER
# =========================================================

def parse_qa_text(text):

    pairs = []

    if not text:
        return pairs

    for line in str(text).splitlines():

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
                {
                    "question": question,
                    "answer": answer
                }
            )

    return pairs


# =========================================================
# TXT Q&A SEARCH
# =========================================================

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

    # Exact match first

    for item in pairs:

        if normalize_text(
            item["question"]
        ) == normalized_question:

            return item["answer"]


    # Fuzzy match

    best_answer = None
    best_score = 0.0

    for item in pairs:

        score = text_similarity(
            question,
            item["question"]
        )

        if score > best_score:

            best_score = score

            best_answer = item["answer"]

    if best_score >= 0.68:

        return best_answer

    return None


# =========================================================
# TEXT CHUNKING
# =========================================================

def split_text(
    text,
    chunk_size=800
):

    if not text:
        return []

    words = str(text).split()

    chunks = []

    current = []
    current_size = 0

    for word in words:

        current.append(word)

        current_size += len(word) + 1

        if current_size >= chunk_size:

            chunks.append(
                " ".join(current)
            )

            current = []
            current_size = 0

    if current:

        chunks.append(
            " ".join(current)
        )

    return chunks


# =========================================================
# DOCUMENT SEARCH
# =========================================================

def search_documents(
    question,
    files
):

    documents = []

    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") not in [
            "txt",
            "pdf",
            "docx"
        ]:
            continue

        text = file.get(
            "text",
            ""
        )

        if not text:
            continue

        chunks = split_text(
            text
        )

        for chunk in chunks:

            documents.append(
                {
                    "file": file.get(
                        "name",
                        "Unknown"
                    ),
                    "text": chunk
                }
            )

    if not documents:
        return []

    corpus = [
        item["text"]
        for item in documents
    ]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            corpus + [question]
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

    for index in ranked[:5]:

        score = float(
            scores[index]
        )

        if score < 0.05:
            continue

        results.append(
            {
                "file": documents[index]["file"],
                "text": documents[index]["text"],
                "score": score
            }
        )

    return results


# =========================================================
# FIND RELEVANT CSV COLUMNS
# =========================================================

def find_relevant_columns(
    question,
    df
):

    q = normalize_text(
        question
    )

    q_no_space = q.replace(
        " ",
        ""
    )

    matches = []

    question_words = set(
        q.split()
    )

    for column in df.columns:

        column_text = normalize_text(
            column
        )

        column_key_value = column_key(
            column
        )

        score = 0.0

        if column_text in q:

            score = 1.0

        elif (
            column_key_value
            and column_key_value in q_no_space
        ):

            score = 0.95

        else:

            score = text_similarity(
                q,
                column_text
            )

        column_words = set(
            column_text.split()
        )

        overlap = len(
            question_words.intersection(
                column_words
            )
        )

        if overlap > 0:

            score = max(
                score,
                0.75
            )

        # Special common names

        aliases = {

            "bp": [
                "blood pressure",
                "bp"
            ],

            "bgr": [
                "blood glucose",
                "blood sugar",
                "glucose"
            ],

            "bu": [
                "blood urea",
                "urea"
            ],

            "sc": [
                "serum creatinine",
                "creatinine"
            ],

            "hemo": [
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

            "salary": [
                "salary"
            ],

            "income": [
                "income"
            ],

            "gpa": [
                "gpa",
                "grade point average"
            ]
        }

        key = column_key_value

        if key in aliases:

            for alias in aliases[key]:

                if normalize_text(
                    alias
                ) in q:

                    score = max(
                        score,
                        0.98
                    )

        if score >= 0.60:

            matches.append(
                (
                    column,
                    score
                )
            )

    matches.sort(
        key=lambda x: x[1],
        reverse=True
    )

    return [
        column
        for column, score in matches[:5]
    ]


# =========================================================
# CSV ANALYSIS
# =========================================================

def analyze_csv_question(
    question,
    df
):

    if df is None:
        return None, []

    if df.empty:

        return (
            "The dataset is empty.",
            []
        )

    q = normalize_text(
        question
    )

    columns = find_relevant_columns(
        question,
        df
    )


    # -----------------------------------------------------
    # TOTAL ROWS
    # -----------------------------------------------------

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "how many records" in q
        or "total records" in q
        or "dataset size" in q
    ):

        return (
            f"This dataset contains "
            f"**{len(df):,} rows**.",
            []
        )


    # -----------------------------------------------------
    # TOTAL COLUMNS
    # -----------------------------------------------------

    if (
        "how many columns" in q
        or "number of columns" in q
        or "total columns" in q
    ):

        return (
            f"This dataset contains "
            f"**{len(df.columns)} columns**.",
            []
        )


    # -----------------------------------------------------
    # COLUMN NAMES
    # -----------------------------------------------------

    if (
        "column names" in q
        or "column name" in q
        or "what are the columns" in q
        or "list columns" in q
        or "show columns" in q
    ):

        names = ", ".join(
            str(column)
            for column in df.columns
        )

        return (
            f"The columns are: **{names}**",
            []
        )


    # No relevant field
    if not columns:
        return None, []


    column = columns[0]


    # -----------------------------------------------------
    # AVERAGE / MEAN
    # -----------------------------------------------------

    if (
        "average" in q
        or "mean" in q
        or "avg" in q
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        value = series.mean()

        if pd.notna(value):

            return (
                f"The average **{column}** is "
                f"**{value:.2f}**.",
                [column]
            )


    # -----------------------------------------------------
    # MAXIMUM
    # -----------------------------------------------------

    if (
        "maximum" in q
        or "max" in q
        or "highest" in q
        or "largest" in q
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        value = series.max()

        if pd.notna(value):

            return (
                f"The maximum **{column}** is "
                f"**{value:.2f}**.",
                [column]
            )


    # -----------------------------------------------------
    # MINIMUM
    # -----------------------------------------------------

    if (
        "minimum" in q
        or "min" in q
        or "lowest" in q
        or "smallest" in q
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        value = series.min()

        if pd.notna(value):

            return (
                f"The minimum **{column}** is "
                f"**{value:.2f}**.",
                [column]
            )


    # -----------------------------------------------------
    # MEDIAN
    # -----------------------------------------------------

    if "median" in q:

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        value = series.median()

        if pd.notna(value):

            return (
                f"The median **{column}** is "
                f"**{value:.2f}**.",
                [column]
            )


    # -----------------------------------------------------
    # STANDARD DEVIATION
    # -----------------------------------------------------

    if (
        "standard deviation" in q
        or "std" in q
    ):

        series = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        value = series.std()

        if pd.notna(value):

            return (
                f"The standard deviation of "
                f"**{column}** is "
                f"**{value:.2f}**.",
                [column]
            )


    # -----------------------------------------------------
    # UNIQUE VALUES
    # -----------------------------------------------------

    if (
        "unique" in q
        or "distinct" in q
    ):

        count = df[column].nunique(
            dropna=True
        )

        return (
            f"**{column}** has "
            f"**{count:,} unique values**.",
            [column]
        )


    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    if (
        "missing" in q
        or "null" in q
        or "empty" in q
        or "nan" in q
    ):

        count = int(
            df[column].isna().sum()
        )

        return (
            f"**{column}** has "
            f"**{count:,} missing values**.",
            [column]
        )


    # -----------------------------------------------------
    # DATA TYPE
    # -----------------------------------------------------

    if (
        "data type" in q
        or "datatype" in q
        or "type of" in q
    ):

        dtype = str(
            df[column].dtype
        )

        return (
            f"The data type of "
            f"**{column}** is "
            f"**{dtype}**.",
            [column]
        )


    # -----------------------------------------------------
    # COLUMN MEANING
    # -----------------------------------------------------

    if (
        "what is" in q
        or "what does" in q
        or "meaning" in q
        or "means" in q
        or "define" in q
    ):

        full_name, meaning = (
            get_column_info(
                column
            )
        )

        return (
            f"**{column}** means "
            f"**{full_name}**. "
            f"{meaning}.",
            [column]
        )


    # -----------------------------------------------------
    # FIRST / LAST VALUE
    # -----------------------------------------------------

    if (
        "first value" in q
        or "first data" in q
    ):

        value = df[column].iloc[0]

        return (
            f"The first value of "
            f"**{column}** is "
            f"**{value}**.",
            [column]
        )


    if (
        "last value" in q
        or "last data" in q
    ):

        value = df[column].iloc[-1]

        return (
            f"The last value of "
            f"**{column}** is "
            f"**{value}**.",
            [column]
        )


    return None, []


# =========================================================
# GENERAL ANSWERS
# =========================================================

GENERAL_ANSWERS = {

    "what is ai":
        "AI (Artificial Intelligence) is a technology that allows computers to perform tasks that normally require human intelligence.",

    "what is artificial intelligence":
        "Artificial Intelligence is a field of computer science that enables machines to perform tasks such as learning, reasoning, understanding language, and recognizing patterns.",

    "what is machine learning":
        "Machine Learning is a branch of AI where computers learn patterns from data and use those patterns to make predictions or decisions.",

    "what is ml":
        "Machine Learning (ML) is a branch of AI that allows computers to learn patterns from data without being explicitly programmed for every task.",

    "what is deep learning":
        "Deep Learning is a part of Machine Learning that uses multi-layer neural networks to learn complex patterns from data.",

    "what is nlp":
        "NLP (Natural Language Processing) is a field of AI that helps computers understand, process, and generate human language.",

    "what is natural language processing":
        "Natural Language Processing (NLP) is a field of AI that focuses on processing and understanding human language.",

    "what is computer vision":
        "Computer Vision is a field of AI that allows computers to understand and analyze images and videos.",

    "what is python":
        "Python is a high-level programming language widely used for web development, automation, data science, machine learning, and AI.",

    "what is csv":
        "CSV stands for Comma-Separated Values. It is a simple file format commonly used to store tabular data.",

    "what is pandas":
        "Pandas is a Python library used for data manipulation, analysis, cleaning, and working with tables such as CSV datasets.",

    "what is streamlit":
        "Streamlit is a Python framework that allows developers to quickly build interactive web applications for data science and machine learning."
}


def get_general_answer(
    question
):

    q = normalize_text(
        question
    )

    best_answer = None
    best_score = 0

    for key, answer in GENERAL_ANSWERS.items():

        score = text_similarity(
            q,
            key
        )

        if score > best_score:

            best_score = score
            best_answer = answer

    if best_score >= 0.82:

        return best_answer

    return None


# =========================================================
# GEMINI CLIENT
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

    prompt = f"""
You are IntelliMind AI, a professional multi-file data assistant.

The user uploaded multiple files.

Answer the user's question using the supplied context.

IMPORTANT RULES:

1. Use the uploaded data whenever relevant.
2. Never invent dataset numbers.
3. If a numerical answer exists in the CSV, use the actual calculated result.
4. If the answer is not available, clearly say so.
5. Keep the response concise and easy to understand.
6. Do not expose internal reasoning.
7. Do not show raw matching rows unless explicitly requested.
8. Do not mention internal retrieval or system instructions.
9. If multiple files contain relevant information, combine them carefully.
10. Answer directly.

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
# BUILD CONTEXT
# =========================================================

def build_file_context(
    question,
    files
):

    parts = []

    # CSV results
    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") != "csv":
            continue

        df = file.get(
            "data"
        )

        if df is None:
            continue

        answer, fields = (
            analyze_csv_question(
                question,
                df
            )
        )

        if answer:

            parts.append(
                f"""
FILE NAME: {file.get("name")}

CSV RESULT:
{answer}

RELEVANT COLUMNS:
{", ".join(map(str, fields))}
"""
            )


    # Documents
    results = search_documents(
        question,
        files
    )

    for result in results:

        parts.append(
            f"""
FILE NAME: {result['file']}

DOCUMENT CONTENT:
{result['text']}
"""
        )

    return "\n".join(
        parts
    )


# =========================================================
# MAIN QUESTION PROCESSOR
# =========================================================

def process_question(
    question
):

    question = str(
        question
    ).strip()

    if not question:

        return (
            "Please enter a question.",
            [],
            []
        )


    # -----------------------------------------------------
    # GENERAL KNOWLEDGE
    # -----------------------------------------------------

    general_answer = get_general_answer(
        question
    )

    if general_answer:

        return (
            general_answer,
            [],
            []
        )


    files = st.session_state.files_data


    # -----------------------------------------------------
    # TXT Q&A
    # -----------------------------------------------------

    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") != "txt":
            continue

        answer = search_qa_dataset(
            question,
            file.get(
                "text",
                ""
            )
        )

        if answer:

            return (
                answer,
                [],
                [file.get(
                    "name",
                    "TXT"
                )]
            )


    # -----------------------------------------------------
    # DIRECT CSV ANALYSIS
    # -----------------------------------------------------

    csv_results = []

    all_fields = []

    source_files = []

    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") != "csv":
            continue

        df = file.get(
            "data"
        )

        if df is None:
            continue

        answer, fields = (
            analyze_csv_question(
                question,
                df
            )
        )

        if answer:

            csv_results.append(
                answer
            )

            all_fields.extend(
                fields
            )

            source_files.append(
                file.get(
                    "name",
                    "CSV"
                )
            )


    # One CSV answer

    if len(csv_results) == 1:

        return (
            csv_results[0],
            list(dict.fromkeys(
                all_fields
            )),
            source_files
        )


    # Multiple CSV answers

    if len(csv_results) > 1:

        combined = "\n\n".join(
            csv_results
        )

        return (
            combined,
            list(dict.fromkeys(
                all_fields
            )),
            source_files
        )


    # -----------------------------------------------------
    # DOCUMENT CONTEXT
    # -----------------------------------------------------

    context = build_file_context(
        question,
        files
    )


    # -----------------------------------------------------
    # GEMINI
    # -----------------------------------------------------

    if context:

        ai_answer = generate_ai_answer(
            question,
            context
        )

        if ai_answer:

            sources = []

            for file in files:

                if not isinstance(
                    file,
                    dict
                ):
                    continue

                filename = file.get(
                    "name",
                    ""
                )

                if filename and filename in context:

                    sources.append(
                        filename
                    )

            return (
                ai_answer,
                [],
                list(dict.fromkeys(
                    sources
                ))
            )


    # -----------------------------------------------------
    # NO ANSWER
    # -----------------------------------------------------

    if not files:

        return (
            "Please upload a CSV, TXT, PDF, or DOCX file first.",
            [],
            []
        )


    return (
        "I could not find a reliable answer "
        "from the uploaded files. Try asking "
        "about a specific column, value, topic, "
        "or document content.",
        [],
        []
    )


# =========================================================
# SMART QUESTIONS
# =========================================================

def generate_smart_questions(
    files
):

    questions = []

    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") != "csv":
            continue

        df = file.get(
            "data"
        )

        if df is None:
            continue

        numeric_columns = (
            df.select_dtypes(
                include=np.number
            ).columns.tolist()
        )

        # Average
        for column in numeric_columns[:3]:

            questions.append(
                f"What is the average {column}?"
            )

        # Max
        if numeric_columns:

            questions.append(
                f"What is the maximum {numeric_columns[0]}?"
            )

        # Rows
        questions.append(
            f"How many rows are in {file.get('name', 'this dataset')}?"
        )

        break


    # Document question

    for file in files:

        if not isinstance(
            file,
            dict
        ):
            continue

        if file.get("type") in [
            "txt",
            "pdf",
            "docx"
        ]:

            questions.append(
                "What is the main information in this document?"
            )

            break


    # Remove duplicates

    final_questions = []

    for question in questions:

        if question not in final_questions:

            final_questions.append(
                question
            )

    return final_questions[:6]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "# 🧠 IntelliMind AI"
    )

    st.caption(
        "Multi-File Data Copilot"
    )

    st.divider()


    # -----------------------------------------------------
    # UPLOAD
    # -----------------------------------------------------

    st.markdown(
        "### 📂 Upload Files"
    )

    uploaded_files = st.file_uploader(
        "Upload CSV, TXT, PDF or DOCX",
        type=[
            "csv",
            "txt",
            "pdf",
            "docx"
        ],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )


    # -----------------------------------------------------
    # PROCESS FILES
    # -----------------------------------------------------

    if uploaded_files:

        current_names = [
            file.name
            for file in uploaded_files
        ]

        if (
            current_names
            != st.session_state.last_uploaded_names
        ):

            st.session_state.files_data = (
                process_uploaded_files(
                    uploaded_files
                )
            )

            st.session_state.last_uploaded_names = (
                current_names
            )


    st.divider()


    # -----------------------------------------------------
    # FILE EXPLORER
    # -----------------------------------------------------

    st.markdown(
        "### 📁 File Explorer"
    )

    valid_files = []

    for item in st.session_state.files_data:

        if isinstance(
            item,
            dict
        ):

            if "name" in item and "type" in item:

                valid_files.append(
                    item
                )


    if valid_files:

        for file_index, file in enumerate(
            valid_files
        ):

            file_type = file.get(
                "type",
                "unknown"
            )

            file_name = file.get(
                "name",
                "Unknown File"
            )


            # Icon

            if file_type == "csv":

                icon = "📊"

            elif file_type == "txt":

                icon = "📝"

            elif file_type == "pdf":

                icon = "📕"

            elif file_type == "docx":

                icon = "📘"

            else:

                icon = "⚠️"


            # ------------------------------------------------
            # CLICKABLE FILE
            # ------------------------------------------------

            with st.expander(
                f"{icon} {file_name}",
                expanded=False
            ):

                # CSV
                if file_type == "csv":

                    df = file.get(
                        "data"
                    )

                    if df is not None:

                        st.caption(
                            f"{len(df):,} rows • "
                            f"{len(df.columns)} columns"
                        )

                        st.markdown(
                            "#### 🧩 All Columns"
                        )

                        for column_index, column in enumerate(
                            df.columns,
                            start=1
                        ):

                            full_name, meaning = (
                                get_column_info(
                                    column
                                )
                            )

                            st.markdown(
                                f"**{column_index}. `{column}`**"
                            )

                            st.caption(
                                f"Full name: {full_name}"
                            )

                            with st.expander(
                                "Meaning",
                                expanded=False
                            ):

                                st.write(
                                    meaning
                                )


                # ------------------------------------------------
                # TXT
                # ------------------------------------------------

                elif file_type == "txt":

                    text_content = file.get(
                        "text",
                        ""
                    )

                    qa_pairs = parse_qa_text(
                        text_content
                    )

                    st.caption(
                        f"{len(qa_pairs):,} Q&A entries"
                    )

                    st.markdown(
                        "#### 📝 File Preview"
                    )

                    if text_content:

                        st.text(
                            text_content[:2500]
                        )

                        if len(
                            text_content
                        ) > 2500:

                            st.caption(
                                "Preview only. The complete file is searchable."
                            )

                    else:

                        st.warning(
                            "No text found."
                        )


                # ------------------------------------------------
                # PDF
                # ------------------------------------------------

                elif file_type == "pdf":

                    text_content = file.get(
                        "text",
                        ""
                    )

                    word_count = len(
                        text_content.split()
                    )

                    st.caption(
                        f"{word_count:,} words extracted"
                    )

                    if text_content:

                        st.markdown(
                            "#### 📄 File Preview"
                        )

                        st.text(
                            text_content[:2500]
                        )

                        if len(
                            text_content
                        ) > 2500:

                            st.caption(
                                "Preview only. The complete PDF text is searchable."
                            )

                    else:

                        st.warning(
                            "No readable text found in this PDF."
                        )


                # ------------------------------------------------
                # DOCX
                # ------------------------------------------------

                elif file_type == "docx":

                    text_content = file.get(
                        "text",
                        ""
                    )

                    word_count = len(
                        text_content.split()
                    )

                    st.caption(
                        f"{word_count:,} words extracted"
                    )

                    if text_content:

                        st.markdown(
                            "#### 📘 File Preview"
                        )

                        st.text(
                            text_content[:2500]
                        )

                        if len(
                            text_content
                        ) > 2500:

                            st.caption(
                                "Preview only. The complete document is searchable."
                            )

                    else:

                        st.warning(
                            "No readable text found in this DOCX."
                        )


                # ------------------------------------------------
                # ERROR
                # ------------------------------------------------

                elif file_type == "error":

                    st.error(
                        file.get(
                            "text",
                            "Unable to read this file."
                        )
                    )


    else:

        st.info(
            "Upload files to explore them here."
        )


    st.divider()


    # -----------------------------------------------------
    # FIELD DICTIONARY
    # -----------------------------------------------------

    if valid_files:

        st.markdown(
            "### 📚 Fields"
        )

        all_columns = []

        for file in valid_files:

            if file.get("type") != "csv":
                continue

            columns = file.get(
                "columns",
                []
            )

            for column in columns:

                if column not in all_columns:

                    all_columns.append(
                        column
                    )


        for column in all_columns[:8]:

            full_name, _ = (
                get_column_info(
                    column
                )
            )

            st.markdown(
                f"`{column}` → {full_name}"
            )


        if len(all_columns) > 8:

            with st.expander(
                "View all fields"
            ):

                for column in all_columns[8:]:

                    full_name, _ = (
                        get_column_info(
                            column
                        )
                    )

                    st.markdown(
                        f"`{column}` → {full_name}"
                    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🧠 IntelliMind AI"
)

st.caption(
    "Analyze CSV datasets and documents together in one intelligent workspace."
)


# =========================================================
# METRICS
# =========================================================

valid_files = [
    item
    for item in st.session_state.files_data
    if isinstance(item, dict)
]


file_count = len(
    valid_files
)

csv_count = sum(
    1
    for item in valid_files
    if item.get("type") == "csv"
)

document_count = sum(
    1
    for item in valid_files
    if item.get("type") in [
        "txt",
        "pdf",
        "docx"
    ]
)

total_rows = sum(
    len(item.get("data"))
    for item in valid_files
    if item.get("type") == "csv"
    and item.get("data") is not None
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "📂 Files",
        file_count
    )


with metric2:

    st.metric(
        "📊 CSV Datasets",
        csv_count
    )


with metric3:

    st.metric(
        "📄 Documents",
        document_count
    )


with metric4:

    st.metric(
        "🔢 Total Rows",
        f"{total_rows:,}"
    )


# =========================================================
# SMART QUESTIONS
# =========================================================

if valid_files:

    st.divider()

    st.subheader(
        "✨ Smart Questions"
    )

    smart_questions = generate_smart_questions(
        valid_files
    )

    if smart_questions:

        smart_columns = st.columns(3)

        for index, question in enumerate(
            smart_questions
        ):

            with smart_columns[
                index % 3
            ]:

                if st.button(
                    question,
                    key=f"smart_question_{index}",
                    use_container_width=True
                ):

                    answer, fields, sources = (
                        process_question(
                            question
                        )
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "user",
                            "content": question
                        }
                    )

                    st.session_state.chat_history.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "fields": fields,
                            "sources": sources
                        }
                    )

                    st.rerun()


# =========================================================
# CHAT
# =========================================================

st.divider()

st.subheader(
    "💬 Conversation"
)


# ---------------------------------------------------------
# OLD CHAT
# ---------------------------------------------------------

for message in st.session_state.chat_history:

    if not isinstance(
        message,
        dict
    ):
        continue

    role = message.get(
        "role"
    )

    content = message.get(
        "content",
        ""
    )


    if role == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                content
            )


    elif role == "assistant":

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                content
            )


            fields = message.get(
                "fields",
                []
            )

            sources = message.get(
                "sources",
                []
            )


            # Relevant fields

            if fields:

                field_info = []

                for field in fields:

                    full_name, _ = (
                        get_column_info(
                            field
                        )
                    )

                    field_info.append(
                        f"`{field}` → {full_name}"
                    )

                st.caption(
                    " • ".join(
                        field_info
                    )
                )


            # Source

            if sources:

                st.caption(
                    "Source: "
                    + ", ".join(
                        sources
                    )
                )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask anything about your uploaded files..."
)


if question:

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    answer, fields, sources = (
        process_question(
            question
        )
    )

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "fields": fields,
            "sources": sources
        }
    )

    st.rerun()
