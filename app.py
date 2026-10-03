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

from google import genai


# =========================================================
# CONFIG
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


# =========================================================
# KNOWN COLUMN FULL FORMS
# =========================================================

COLUMN_INFO = {

    "id": (
        "ID",
        "A unique identification number."
    ),

    "age": (
        "Age",
        "Age of a person, usually measured in years."
    ),

    "bp": (
        "Blood Pressure",
        "Measurement of blood pressure."
    ),

    "bmi": (
        "Body Mass Index",
        "A measure calculated using body weight and height."
    ),

    "dob": (
        "Date of Birth",
        "The date on which a person was born."
    ),

    "gpa": (
        "Grade Point Average",
        "A numerical measure of academic performance."
    ),

    "cgpa": (
        "Cumulative Grade Point Average",
        "Overall academic performance across completed semesters."
    ),

    "ai": (
        "Artificial Intelligence",
        "Technology that enables computers to perform intelligent tasks."
    ),

    "ml": (
        "Machine Learning",
        "A method where computers learn patterns from data."
    ),

    "dl": (
        "Deep Learning",
        "A type of machine learning based on multi-layer neural networks."
    ),

    "nlp": (
        "Natural Language Processing",
        "Technology for processing and understanding human language."
    ),

    "cv": (
        "Computer Vision",
        "Technology that enables computers to understand images and videos."
    ),

    "salary": (
        "Salary",
        "The amount of money earned by an employee."
    ),

    "income": (
        "Income",
        "Money earned by a person or organization."
    ),

    "annualincome": (
        "Annual Income",
        "Total income earned during one year."
    ),

    "experience": (
        "Experience",
        "Amount of professional or practical experience."
    ),

    "department": (
        "Department",
        "The department or organizational unit associated with a record."
    ),

    "name": (
        "Name",
        "Name of the person, object, or entity."
    ),

    "email": (
        "Email Address",
        "Electronic mail address used to identify or contact a person."
    ),

    "phone": (
        "Phone Number",
        "Telephone number used to contact a person."
    ),

    "address": (
        "Address",
        "Location or contact address."
    ),

    "gender": (
        "Gender",
        "Gender information associated with a record."
    ),

    "status": (
        "Status",
        "Current state or condition of a record."
    ),

    "score": (
        "Score",
        "A numerical value representing performance or measurement."
    ),

    "rating": (
        "Rating",
        "A value representing an evaluation or assessment."
    ),

    "price": (
        "Price",
        "Amount of money required to purchase something."
    ),

    "quantity": (
        "Quantity",
        "Number or amount of items."
    ),

    "height": (
        "Height",
        "Measurement of vertical size."
    ),

    "weight": (
        "Weight",
        "Measurement of how heavy something or someone is."
    ),

    "glucose": (
        "Plasma Glucose Concentration",
        "Blood glucose level measured during a test."
    ),

    "bloodpressure": (
        "Blood Pressure",
        "Blood pressure measurement."
    ),

    "pregnancies": (
        "Number of Pregnancies",
        "Number of times a patient has been pregnant."
    ),

    "insulin": (
        "Serum Insulin",
        "Amount of insulin measured in the blood."
    ),

    "outcome": (
        "Outcome",
        "The final result or target value of a record."
    ),

    "sc": (
        "Serum Creatinine",
        "Creatinine level measured in the blood."
    ),

    "bu": (
        "Blood Urea",
        "Amount of urea present in the blood."
    ),

    "hemo": (
        "Hemoglobin",
        "Hemoglobin level in the blood."
    ),

    "pcv": (
        "Packed Cell Volume",
        "Percentage of blood volume occupied by red blood cells."
    ),

    "sod": (
        "Sodium",
        "Sodium level in the blood."
    ),

    "pot": (
        "Potassium",
        "Potassium level in the blood."
    ),

    "htn": (
        "Hypertension",
        "Indicates high blood pressure."
    ),

    "dm": (
        "Diabetes Mellitus",
        "Indicates diabetes mellitus."
    ),

    "cad": (
        "Coronary Artery Disease",
        "A condition affecting arteries supplying the heart."
    ),

    "classification": (
        "Disease Classification",
        "Category or class assigned to a record."
    ),

    "performance_score": (
        "Performance Score",
        "A numerical measurement of performance."
    ),

    "credit_score": (
        "Credit Score",
        "A numerical measure related to creditworthiness."
    ),

    "loan_status": (
        "Loan Status",
        "Current status or result of a loan."
    )
}


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

def similarity(a, b):

    return difflib.SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b)
    ).ratio()


# =========================================================
# CLEAN COLUMN NAME
# =========================================================

def clean_column_name(column):

    column = str(column).strip()

    column = re.sub(
        r"[_\-]+",
        " ",
        column
    )

    column = re.sub(
        r"([a-z])([A-Z])",
        r"\1 \2",
        column
    )

    return column.strip()


# =========================================================
# COLUMN INFORMATION
# =========================================================

def get_column_info(column):

    original = str(column).strip()

    key = normalize_text(
        original
    ).replace(" ", "")

    # Exact known information
    if key in COLUMN_INFO:

        return COLUMN_INFO[key]

    # Similar known information
    best_key = None
    best_score = 0

    for known_key in COLUMN_INFO:

        score = similarity(
            key,
            known_key
        )

        if score > best_score:

            best_score = score
            best_key = known_key

    if best_score >= 0.82:

        return COLUMN_INFO[best_key]

    # Generic fallback
    readable = clean_column_name(
        original
    )

    full_form = readable.title()

    meaning = (
        f"This field contains information about "
        f"{readable.lower()}."
    )

    return full_form, meaning


# =========================================================
# FILE READERS
# =========================================================

def read_csv_file(uploaded_file):

    try:

        return pd.read_csv(
            uploaded_file
        )

    except Exception:

        uploaded_file.seek(0)

        try:

            return pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

        except Exception:

            uploaded_file.seek(0)

            return pd.read_csv(
                uploaded_file,
                encoding="cp1252"
            )


def read_txt_file(uploaded_file):

    raw = uploaded_file.read()

    for encoding in [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]:

        try:

            return raw.decode(
                encoding
            )

        except Exception:
            pass

    return raw.decode(
        "utf-8",
        errors="ignore"
    )


def read_pdf_file(uploaded_file):

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

    return "\n".join(
        pages
    )


def read_docx_file(uploaded_file):

    document = Document(
        uploaded_file
    )

    parts = []

    # Paragraphs
    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            parts.append(text)

    # Tables
    for table in document.tables:

        for row in table.rows:

            values = []

            for cell in row.cells:

                values.append(
                    cell.text.strip()
                )

            parts.append(
                " | ".join(values)
            )

    return "\n".join(
        parts
    )


# =========================================================
# DISPLAY COLUMN DICTIONARY
# =========================================================

def display_column_dictionary(columns):

    if not columns:

        st.info(
            "No columns or fields detected."
        )

        return

    for column in columns:

        full_form, meaning = get_column_info(
            column
        )

        st.markdown(
            f"**`{column}`**"
        )

        st.markdown(
            f"→ **{full_form}**"
        )

        st.caption(
            meaning
        )

        st.divider()


# =========================================================
# DATASET QUESTION GENERATOR
# =========================================================

def generate_dataset_questions(df):

    questions = []

    columns = list(
        df.columns
    )

    numeric_columns = list(
        df.select_dtypes(
            include=np.number
        ).columns
    )

    categorical_columns = list(
        df.select_dtypes(
            include=[
                "object",
                "category",
                "bool"
            ]
        ).columns
    )

    # -----------------------------------------
    # GENERAL QUESTIONS
    # -----------------------------------------

    questions.append(
        "How many rows are in this dataset?"
    )

    questions.append(
        "How many columns are in this dataset?"
    )

    questions.append(
        "Are there any missing values?"
    )

    questions.append(
        "What are the columns in this dataset?"
    )

    # -----------------------------------------
    # NUMERIC QUESTIONS
    # -----------------------------------------

    for column in numeric_columns[:8]:

        readable = clean_column_name(
            column
        )

        questions.append(
            f"What is the average {readable}?"
        )

        questions.append(
            f"What is the highest {readable}?"
        )

        questions.append(
            f"What is the lowest {readable}?"
        )

    # -----------------------------------------
    # CATEGORICAL QUESTIONS
    # -----------------------------------------

    for column in categorical_columns[:6]:

        readable = clean_column_name(
            column
        )

        questions.append(
            f"What are the unique {readable} values?"
        )

        questions.append(
            f"What is the most common {readable}?"
        )

    # -----------------------------------------
    # TARGET-LIKE COLUMN
    # -----------------------------------------

    target_words = [
        "target",
        "outcome",
        "result",
        "class",
        "classification",
        "label",
        "status"
    ]

    for column in columns:

        column_lower = normalize_text(
            column
        )

        if any(
            word in column_lower
            for word in target_words
        ):

            readable = clean_column_name(
                column
            )

            questions.append(
                f"What is the distribution of {readable}?"
            )

    # Remove duplicates
    unique_questions = []

    for question in questions:

        if question not in unique_questions:

            unique_questions.append(
                question
            )

    return unique_questions[:18]


# =========================================================
# DOCUMENT QUESTION SUGGESTIONS
# =========================================================

def generate_document_questions(
    text,
    file_type
):

    questions = [

        "What is this document about?",

        "Give me a short summary of this document.",

        "What are the main points?",

        "What are the important topics in this document?",

        "What important information can you find in this file?",

        "Give me the key information from this document."
    ]

    # Q&A dataset
    if "|" in text:

        questions.insert(
            0,
            "What questions can this file answer?"
        )

    # Tables
    if "|" in text:

        questions.append(
            "What information is available in the tables?"
        )

    return questions


# =========================================================
# CSV COLUMN MATCHING
# =========================================================

def find_relevant_columns(
    question,
    df
):

    q = normalize_text(
        question
    )

    found = []

    for column in df.columns:

        column_text = normalize_text(
            column
        )

        # Direct match
        if column_text in q:

            found.append(
                column
            )

            continue

        # Clean version
        clean = column_text.replace(
            " ",
            ""
        )

        # Known synonyms
        synonyms = {

            "bloodpressure": [
                "blood pressure",
                "bp",
                "pressure"
            ],

            "glucose": [
                "glucose",
                "blood sugar",
                "blood glucose"
            ],

            "salary": [
                "salary",
                "pay",
                "income"
            ],

            "age": [
                "age"
            ],

            "bmi": [
                "bmi",
                "body mass index"
            ],

            "experience": [
                "experience",
                "years experience"
            ]
        }

        if clean in synonyms:

            for synonym in synonyms[clean]:

                if normalize_text(
                    synonym
                ) in q:

                    found.append(
                        column
                    )

                    break

    return list(
        dict.fromkeys(found)
    )


# =========================================================
# DIRECT CSV ANALYSIS
# =========================================================

def direct_csv_analysis(
    question,
    df
):

    q = normalize_text(
        question
    )

    columns = find_relevant_columns(
        question,
        df
    )

    # -----------------------------------------
    # ROW COUNT
    # -----------------------------------------

    if any(
        x in q
        for x in [
            "how many rows",
            "number of rows",
            "total rows",
            "row count",
            "how many records",
            "number of records",
            "total records",
            "how many data"
        ]
    ):

        return (
            f"The dataset contains **{len(df)} rows**.",
            []
        )

    # -----------------------------------------
    # COLUMN COUNT
    # -----------------------------------------

    if any(
        x in q
        for x in [
            "how many columns",
            "number of columns",
            "total columns",
            "column count"
        ]
    ):

        return (
            f"The dataset contains **{len(df.columns)} columns**.",
            []
        )

    # -----------------------------------------
    # COLUMN NAMES
    # -----------------------------------------

    if any(
        x in q
        for x in [
            "column names",
            "what columns",
            "show columns",
            "list columns",
            "what are the columns"
        ]
    ):

        names = ", ".join(
            str(c)
            for c in df.columns
        )

        return (
            f"The columns are: **{names}**",
            list(df.columns)
        )

    # -----------------------------------------
    # SHAPE
    # -----------------------------------------

    if "shape" in q:

        return (
            f"The dataset shape is "
            f"**{df.shape[0]} rows × {df.shape[1]} columns**.",
            []
        )

    # -----------------------------------------
    # MISSING
    # -----------------------------------------

    if any(
        x in q
        for x in [
            "missing",
            "null",
            "empty values"
        ]
    ):

        missing = int(
            df.isnull().sum().sum()
        )

        return (
            f"The dataset contains "
            f"**{missing} missing values**.",
            []
        )

    # -----------------------------------------
    # COLUMN MEANING
    # -----------------------------------------

    if columns and any(
        x in q
        for x in [
            "full form",
            "meaning",
            "what does",
            "stands for"
        ]
    ):

        column = columns[0]

        full_form, meaning = get_column_info(
            column
        )

        return (
            f"**{column}** → **{full_form}**\n\n"
            f"{meaning}",
            columns
        )

    # -----------------------------------------
    # NUMERIC ANALYSIS
    # -----------------------------------------

    if columns:

        column = columns[0]

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(numeric) == 0:

            # Categorical data
            unique_values = df[
                column
            ].dropna().unique()

            if any(
                x in q
                for x in [
                    "unique",
                    "values",
                    "categories"
                ]
            ):

                values = ", ".join(
                    str(x)
                    for x in unique_values[:30]
                )

                return (
                    f"Unique values in **{column}**: "
                    f"**{values}**",
                    columns
                )

            if any(
                x in q
                for x in [
                    "most common",
                    "common",
                    "frequent"
                ]
            ):

                mode = (
                    df[column]
                    .dropna()
                    .mode()
                )

                if not mode.empty:

                    return (
                        f"The most common "
                        f"**{column}** is "
                        f"**{mode.iloc[0]}**.",
                        columns
                    )

            return None, columns

        # Average
        if any(
            x in q
            for x in [
                "average",
                "avg",
                "mean",
                "avarage"
            ]
        ):

            value = numeric.mean()

            return (
                f"The average **{column}** is "
                f"**{value:.2f}**.",
                columns
            )

        # Maximum
        if any(
            x in q
            for x in [
                "highest",
                "maximum",
                "max",
                "largest"
            ]
        ):

            value = numeric.max()

            return (
                f"The highest **{column}** value is "
                f"**{value:.2f}**.",
                columns
            )

        # Minimum
        if any(
            x in q
            for x in [
                "lowest",
                "minimum",
                "min",
                "smallest"
            ]
        ):

            value = numeric.min()

            return (
                f"The lowest **{column}** value is "
                f"**{value:.2f}**.",
                columns
            )

        # Median
        if "median" in q:

            value = numeric.median()

            return (
                f"The median **{column}** is "
                f"**{value:.2f}**.",
                columns
            )

        # Sum
        if any(
            x in q
            for x in [
                "sum",
                "total"
            ]
        ):

            value = numeric.sum()

            return (
                f"The total **{column}** is "
                f"**{value:.2f}**.",
                columns
            )

    return None, columns


# =========================================================
# TXT Q&A SEARCH
# =========================================================

def search_qa_dataset(
    question,
    text
):

    entries = []

    for line in text.splitlines():

        if "|" not in line:
            continue

        q, a = line.split(
            "|",
            1
        )

        q = q.strip()
        a = a.strip()

        if q and a:

            entries.append(
                (q, a)
            )

    if not entries:

        return None

    normalized_question = normalize_text(
        question
    )

    # Exact
    for stored_q, answer in entries:

        if normalize_text(
            stored_q
        ) == normalized_question:

            return answer

    # Fuzzy
    best_score = 0
    best_answer = None

    for stored_q, answer in entries:

        score = similarity(
            question,
            stored_q
        )

        if score > best_score:

            best_score = score
            best_answer = answer

    if best_score >= 0.68:

        return best_answer

    return None


# =========================================================
# GENERIC TEXT SEARCH
# =========================================================

def split_text(
    text,
    chunk_size=800
):

    words = text.split()

    chunks = []

    current = []
    size = 0

    for word in words:

        current.append(
            word
        )

        size += len(word)

        if size >= chunk_size:

            chunks.append(
                " ".join(current)
            )

            current = []
            size = 0

    if current:

        chunks.append(
            " ".join(current)
        )

    return chunks


def search_text(
    question,
    text
):

    chunks = split_text(
        text
    )

    if not chunks:

        return None

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            chunks
        )

        query_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            matrix
        )[0]

        index = int(
            np.argmax(scores)
        )

        if scores[index] >= 0.12:

            return chunks[index]

    except Exception:

        pass

    return None


# =========================================================
# LOCAL ANSWERS
# =========================================================

def local_answer(question):

    q = normalize_text(
        question
    )

    if q in [
        "hi",
        "hello",
        "hey",
        "hii",
        "helo"
    ]:

        return (
            "Hello! 👋 How can I help you?"
        )

    if "how are you" in q:

        return (
            "I'm doing great! 😊 "
            "How can I help you?"
        )

    if "who are you" in q:

        return (
            "I'm **IntelliMind AI**, an AI-powered "
            "file and dataset analysis assistant."
        )

    if "what can you do" in q:

        return (
            "I can analyze CSV datasets, read TXT/PDF/Word "
            "files, answer questions, explain fields, "
            "and suggest useful questions."
        )

    if q in [
        "thanks",
        "thank you"
    ]:

        return "You're welcome! 😊"

    return None


# =========================================================
# BUILT-IN AI KNOWLEDGE
# =========================================================

def built_in_answer(question):

    q = normalize_text(
        question
    )

    answers = {

        "ai":
            "**AI** stands for **Artificial Intelligence**. "
            "It enables computers to perform tasks that normally "
            "require human intelligence.",

        "ml":
            "**ML** stands for **Machine Learning**. "
            "It allows computers to learn patterns from data.",

        "machine learning":
            "**Machine Learning** is a branch of AI where "
            "computers learn patterns from data to make "
            "predictions or decisions.",

        "dl":
            "**DL** stands for **Deep Learning**. "
            "It uses multi-layer neural networks.",

        "nlp":
            "**NLP** stands for **Natural Language Processing**. "
            "It helps computers understand and process human language.",

        "cv":
            "**CV** stands for **Computer Vision**. "
            "It helps computers understand images and videos.",

        "python":
            "**Python** is a high-level programming language "
            "widely used in AI, machine learning, data science "
            "and software development."
    }

    return answers.get(
        q
    )


# =========================================================
# GEMINI CLIENT
# =========================================================

def get_gemini_client():

    try:

        api_key = st.secrets[
            "GEMINI_API_KEY"
        ]

    except Exception:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:

        return None

    return genai.Client(
        api_key=api_key
    )


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    file_context
):

    client = get_gemini_client()

    if client is None:

        return None

    prompt = f"""
You are IntelliMind AI.

Answer the user's question using the uploaded file information.

USER QUESTION:
{question}

UPLOADED FILE INFORMATION:
{file_context}

RULES:

1. Give a clear and simple answer.

2. If the question is about numerical dataset information,
   use the provided data.

3. Never invent numbers.

4. If the information is not available, say so.

5. Do not show internal processing.

6. Do not show matching rows unless explicitly requested.

7. Do not create a "Relevant Column" table.

8. Keep the answer concise.

9. If the question is general, answer normally.
"""

    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        return response.text

    except Exception:

        return None


# =========================================================
# CREATE CONTEXT
# =========================================================

def create_file_context(question):

    contexts = []

    for filename, data in st.session_state.files_data.items():

        if data["type"] == "csv":

            df = data["data"]

            relevant = find_relevant_columns(
                question,
                df
            )

            if relevant:

                sample = df[
                    relevant
                ].head(20).to_string(
                    index=False
                )

                contexts.append(
                    f"""
FILE: {filename}

COLUMNS:
{relevant}

DATA:
{sample}
"""
                )

        else:

            text = data["data"]

            result = search_text(
                question,
                text
            )

            if result:

                contexts.append(
                    f"""
FILE: {filename}

CONTENT:
{result}
"""
                )

    return "\n\n".join(
        contexts
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 IntelliMind AI")

    st.caption(
        "Smart File & Dataset Assistant"
    )

    st.divider()

    st.subheader("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "CSV, TXT, PDF or Word",
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

            extension = (
                filename.lower()
                .split(".")[-1]
            )

            try:

                if extension == "csv":

                    df = read_csv_file(
                        uploaded_file
                    )

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
                        "data": text
                    }

                elif extension == "pdf":

                    text = read_pdf_file(
                        uploaded_file
                    )

                    st.session_state.files_data[
                        filename
                    ] = {
                        "type": "pdf",
                        "data": text
                    }

                elif extension == "docx":

                    text = read_docx_file(
                        uploaded_file
                    )

                    st.session_state.files_data[
                        filename
                    ] = {
                        "type": "docx",
                        "data": text
                    }

            except Exception as e:

                st.error(
                    f"Error reading {filename}: {e}"
                )

    st.divider()

    # =====================================================
    # UPLOADED FILE LIST
    # =====================================================

    st.subheader("📂 Your Files")

    if st.session_state.files_data:

        for filename, data in st.session_state.files_data.items():

            if data["type"] == "csv":

                df = data["data"]

                st.caption(
                    f"📊 {filename}"
                )

                st.caption(
                    f"{len(df)} rows • "
                    f"{len(df.columns)} columns"
                )

            else:

                st.caption(
                    f"📄 {filename}"
                )

    else:

        st.caption(
            "No files uploaded yet."
        )

    st.divider()

    # =====================================================
    # COLUMN DICTIONARY IN SIDEBAR
    # =====================================================

    st.subheader(
        "📚 Column / Field Dictionary"
    )

    dictionary_found = False

    for filename, data in st.session_state.files_data.items():

        # CSV
        if data["type"] == "csv":

            dictionary_found = True

            df = data["data"]

            st.markdown(
                f"**📊 {filename}**"
            )

            for column in df.columns:

                full_form, meaning = get_column_info(
                    column
                )

                with st.expander(
                    str(column)
                ):

                    st.markdown(
                        f"**Full Form:** "
                        f"{full_form}"
                    )

                    st.caption(
                        meaning
                    )

        # TXT/PDF/DOCX
        else:

            text = data["data"]

            possible_fields = []

            for line in text.splitlines()[:300]:

                line = line.strip()

                if not line:
                    continue

                if ":" in line:

                    first = line.split(
                        ":",
                        1
                    )[0].strip()

                    if (
                        1 <= len(first.split()) <= 6
                        and len(first) <= 50
                    ):

                        possible_fields.append(
                            first
                        )

            if possible_fields:

                dictionary_found = True

                st.markdown(
                    f"**📄 {filename}**"
                )

                for field in list(
                    dict.fromkeys(
                        possible_fields
                    )
                )[:30]:

                    full_form, meaning = get_column_info(
                        field
                    )

                    with st.expander(
                        str(field)
                    ):

                        st.markdown(
                            f"**Full Form:** "
                            f"{full_form}"
                        )

                        st.caption(
                            meaning
                        )

    if not dictionary_found:

        st.caption(
            "Upload a file to see columns/fields."
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.title(
    "🤖 IntelliMind AI"
)

st.caption(
    "Ask questions about your datasets and documents."
)


# =========================================================
# DATASET OVERVIEW
# =========================================================

if st.session_state.files_data:

    st.subheader(
        "📊 File Overview"
    )

    for filename, data in st.session_state.files_data.items():

        if data["type"] == "csv":

            df = data["data"]

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Rows",
                len(df)
            )

            c2.metric(
                "Columns",
                len(df.columns)
            )

            c3.metric(
                "Numeric",
                len(
                    df.select_dtypes(
                        include=np.number
                    ).columns
                )
            )

            c4.metric(
                "Missing",
                int(
                    df.isnull().sum().sum()
                )
            )

            st.caption(
                f"📊 {filename}"
            )


# =========================================================
# SMART QUESTION SUGGESTIONS
# =========================================================

if st.session_state.files_data:

    st.subheader(
        "💡 Suggested Questions"
    )

    all_questions = []

    for filename, data in st.session_state.files_data.items():

        if data["type"] == "csv":

            suggestions = generate_dataset_questions(
                data["data"]
            )

        else:

            suggestions = generate_document_questions(
                data["data"],
                data["type"]
            )

        for question in suggestions:

            if question not in all_questions:

                all_questions.append(
                    question
                )

    # Show first 8
    visible_questions = all_questions[:8]

    cols = st.columns(2)

    for index, question in enumerate(
        visible_questions
    ):

        with cols[index % 2]:

            if st.button(
                f"💬 {question}",
                key=f"suggest_{index}",
                use_container_width=True
            ):

                st.session_state.selected_question = (
                    question
                )

                st.rerun()


# =========================================================
# SELECTED QUESTION
# =========================================================

if st.session_state.selected_question:

    st.info(
        f"Selected question: "
        f"**{st.session_state.selected_question}**"
    )


# =========================================================
# CHAT HISTORY
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
            and message.get("columns")
        ):

            st.markdown(
                "### 📌 Column Information"
            )

            for column in message["columns"]:

                full_form, meaning = get_column_info(
                    column
                )

                st.markdown(
                    f"**`{column}` → {full_form}**"
                )

                st.caption(
                    meaning
                )


# =========================================================
# CHAT INPUT
# =========================================================

typed_question = st.chat_input(
    "Ask IntelliMind anything about your files..."
)

question = typed_question

if not question and st.session_state.selected_question:

    question = st.session_state.selected_question

    st.session_state.selected_question = ""


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # User message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(
            question
        )

    answer = None
    answer_columns = []

    # =====================================================
    # 1. DIRECT CSV ANALYSIS
    # =====================================================

    for filename, data in st.session_state.files_data.items():

        if data["type"] == "csv":

            result, columns = direct_csv_analysis(
                question,
                data["data"]
            )

            if result:

                answer = result

                answer_columns = columns

                break


    # =====================================================
    # 2. TXT/PDF/DOCX Q&A
    # =====================================================

    if answer is None:

        for filename, data in st.session_state.files_data.items():

            if data["type"] in [
                "txt",
                "pdf",
                "docx"
            ]:

                result = search_qa_dataset(
                    question,
                    data["data"]
                )

                if result:

                    answer = result

                    break


    # =====================================================
    # 3. LOCAL
    # =====================================================

    if answer is None:

        answer = local_answer(
            question
        )


    # =====================================================
    # 4. BUILT-IN
    # =====================================================

    if answer is None:

        answer = built_in_answer(
            question
        )


    # =====================================================
    # 5. GEMINI
    # =====================================================

    if answer is None:

        file_context = create_file_context(
            question
        )

        answer = generate_ai_answer(
            question,
            file_context
        )


    # =====================================================
    # 6. FALLBACK
    # =====================================================

    if answer is None:

        answer = (
            "I couldn't find a reliable answer "
            "from the uploaded files. "
            "Please try a more specific question."
        )


    # =====================================================
    # SHOW ANSWER
    # =====================================================

    with st.chat_message(
        "assistant"
    ):

        st.markdown(
            answer
        )

        if answer_columns:

            st.markdown(
                "### 📌 Column Information"
            )

            for column in answer_columns:

                full_form, meaning = get_column_info(
                    column
                )

                st.markdown(
                    f"**`{column}` → {full_form}**"
                )

                st.caption(
                    meaning
                )


    # =====================================================
    # SAVE
    # =====================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "columns": answer_columns
        }
    )
