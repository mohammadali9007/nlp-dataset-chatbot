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

if "files_data" not in st.session_state:
    st.session_state.files_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# COLUMN / FIELD INFORMATION
# =========================================================

COLUMN_INFO = {

    # ---------------- DIABETES ----------------

    "pregnancies": {
        "full_form": "Number of Pregnancies",
        "meaning": "Number of times the patient has been pregnant."
    },

    "glucose": {
        "full_form": "Plasma Glucose Concentration",
        "meaning": "Blood glucose level measured during the test."
    },

    "bloodpressure": {
        "full_form": "Diastolic Blood Pressure",
        "meaning": "Diastolic blood pressure measured in mm Hg."
    },

    "skinthickness": {
        "full_form": "Triceps Skin Fold Thickness",
        "meaning": "Skin fold thickness measured at the triceps."
    },

    "insulin": {
        "full_form": "Serum Insulin",
        "meaning": "Amount of insulin measured in the blood."
    },

    "bmi": {
        "full_form": "Body Mass Index",
        "meaning": "A measure based on body weight and height."
    },

    "diabetespedigreefunction": {
        "full_form": "Diabetes Pedigree Function",
        "meaning": "A score related to the likelihood of diabetes based on family history."
    },

    "age": {
        "full_form": "Age",
        "meaning": "Age of the person in years."
    },

    "outcome": {
        "full_form": "Diabetes Outcome",
        "meaning": "Usually 1 means diabetic and 0 means non-diabetic."
    },


    # ---------------- KIDNEY ----------------

    "id": {
        "full_form": "Patient ID",
        "meaning": "Unique identification number of the patient."
    },

    "bp": {
        "full_form": "Blood Pressure",
        "meaning": "Blood pressure measurement of the patient."
    },

    "sg": {
        "full_form": "Specific Gravity",
        "meaning": "Specific gravity of urine."
    },

    "al": {
        "full_form": "Albumin",
        "meaning": "Albumin level detected in urine."
    },

    "su": {
        "full_form": "Sugar",
        "meaning": "Sugar level detected in urine."
    },

    "rbc": {
        "full_form": "Red Blood Cells",
        "meaning": "Presence or level of red blood cells."
    },

    "pc": {
        "full_form": "Pus Cell",
        "meaning": "Presence of pus cells in urine."
    },

    "pcc": {
        "full_form": "Pus Cell Clumps",
        "meaning": "Presence of groups or clumps of pus cells."
    },

    "ba": {
        "full_form": "Bacteria",
        "meaning": "Presence of bacteria in the sample."
    },

    "bgr": {
        "full_form": "Blood Glucose Random",
        "meaning": "Random blood glucose level."
    },

    "bu": {
        "full_form": "Blood Urea",
        "meaning": "Amount of urea in the blood."
    },

    "sc": {
        "full_form": "Serum Creatinine",
        "meaning": "Creatinine level in the blood."
    },

    "sod": {
        "full_form": "Sodium",
        "meaning": "Sodium level in the blood."
    },

    "pot": {
        "full_form": "Potassium",
        "meaning": "Potassium level in the blood."
    },

    "hemo": {
        "full_form": "Hemoglobin",
        "meaning": "Hemoglobin level in the blood."
    },

    "pcv": {
        "full_form": "Packed Cell Volume",
        "meaning": "Percentage of blood volume occupied by red blood cells."
    },

    "wc": {
        "full_form": "White Blood Cell Count",
        "meaning": "Number of white blood cells in the blood."
    },

    "rc": {
        "full_form": "Red Blood Cell Count",
        "meaning": "Number of red blood cells in the blood."
    },

    "htn": {
        "full_form": "Hypertension",
        "meaning": "Indicates high blood pressure."
    },

    "dm": {
        "full_form": "Diabetes Mellitus",
        "meaning": "Indicates the presence of diabetes mellitus."
    },

    "cad": {
        "full_form": "Coronary Artery Disease",
        "meaning": "A condition affecting the arteries supplying the heart."
    },

    "appet": {
        "full_form": "Appetite",
        "meaning": "Indicates the patient's appetite."
    },

    "pe": {
        "full_form": "Pedal Edema",
        "meaning": "Swelling caused by fluid accumulation, commonly in the feet or legs."
    },

    "ane": {
        "full_form": "Anemia",
        "meaning": "Indicates whether anemia is present."
    },

    "classification": {
        "full_form": "Disease Classification",
        "meaning": "The final disease classification or category."
    }
}


# =========================================================
# SYNONYMS
# =========================================================

COLUMN_SYNONYMS = {

    "glucose": [
        "glucose",
        "blood sugar",
        "blood glucose",
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

    "bgr": [
        "random blood glucose",
        "blood glucose random",
        "bgr"
    ],

    "pcv": [
        "packed cell volume",
        "pcv"
    ],

    "wc": [
        "white blood cell",
        "white blood cell count",
        "wbc",
        "wc"
    ],

    "rc": [
        "red blood cell",
        "red blood cell count",
        "rbc count",
        "rc"
    ],

    "htn": [
        "hypertension",
        "high blood pressure",
        "htn"
    ],

    "dm": [
        "diabetes mellitus",
        "dm"
    ],

    "cad": [
        "coronary artery disease",
        "cad"
    ],

    "classification": [
        "classification",
        "class",
        "result"
    ]
}


# =========================================================
# NORMALIZE TEXT
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
# FILE READERS
# =========================================================

def read_csv_file(uploaded_file):

    try:
        return pd.read_csv(uploaded_file)

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

    encodings = [
        "utf-8",
        "utf-8-sig",
        "cp1252",
        "latin1"
    ]

    for enc in encodings:

        try:
            return raw.decode(enc)

        except Exception:
            pass

    return raw.decode(
        "utf-8",
        errors="ignore"
    )


def read_pdf_file(uploaded_file):

    reader = PdfReader(uploaded_file)

    pages = []

    for page in reader.pages:

        try:

            text = page.extract_text()

            if text:
                pages.append(text)

        except Exception:
            pass

    return "\n".join(pages)


def read_docx_file(uploaded_file):

    document = Document(uploaded_file)

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
                values.append(cell.text.strip())

            parts.append(" | ".join(values))

    return "\n".join(parts)


# =========================================================
# EXTRACT PDF / WORD TABLE COLUMNS
# =========================================================

def extract_table_columns_from_text(text):

    columns = []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    # Look for pipe-separated table rows
    for line in lines:

        if "|" in line:

            parts = [
                x.strip()
                for x in line.split("|")
                if x.strip()
            ]

            if 2 <= len(parts) <= 20:

                for item in parts:

                    if len(item) <= 50:
                        columns.append(item)

                break

    return list(dict.fromkeys(columns))


# =========================================================
# EXTRACT HEADINGS
# =========================================================

def extract_possible_fields(text):

    fields = []

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    for line in lines[:300]:

        clean = line.strip()

        # Ignore very long sentences
        if len(clean) > 70:
            continue

        # Remove numbering
        clean = re.sub(
            r"^[\d\.\-\_\)\(]+\s*",
            "",
            clean
        )

        # Common heading patterns
        if re.match(
            r"^[A-Za-z][A-Za-z0-9 _\-/]{1,50}:?$",
            clean
        ):

            word_count = len(clean.split())

            if word_count <= 8:

                if clean.lower() not in [
                    "question",
                    "answer",
                    "yes",
                    "no",
                    "hello"
                ]:

                    fields.append(
                        clean.rstrip(":")
                    )

    return list(dict.fromkeys(fields))


# =========================================================
# COLUMN INFO GENERATOR
# =========================================================

def get_column_info(column):

    original = str(column).strip()

    key = normalize_text(
        original
    ).replace(" ", "")

    # Exact known mapping
    if key in COLUMN_INFO:

        return COLUMN_INFO[key]["full_form"], COLUMN_INFO[key]["meaning"]

    # Synonym matching
    for col_key, synonyms in COLUMN_SYNONYMS.items():

        for synonym in synonyms:

            if similarity(
                original,
                synonym
            ) >= 0.82:

                if col_key in COLUMN_INFO:

                    return (
                        COLUMN_INFO[col_key]["full_form"],
                        COLUMN_INFO[col_key]["meaning"]
                    )

    # Acronym fallback
    words = re.findall(
        r"[A-Za-z]+",
        original
    )

    if len(words) > 1:

        full_form = " ".join(
            word.capitalize()
            for word in words
        )

    else:

        full_form = original.replace(
            "_",
            " "
        ).replace(
            "-",
            " "
        ).title()

    meaning = (
        f"This field represents information about "
        f"{original.replace('_', ' ').replace('-', ' ')}."
    )

    return full_form, meaning


# =========================================================
# DISPLAY COLUMN / FIELD INFORMATION
# =========================================================

def show_column_information(columns, title="📚 Column / Field Information"):

    if not columns:
        return

    st.markdown(f"### {title}")

    for column in columns:

        full_form, meaning = get_column_info(
            column
        )

        st.markdown(
            f"**`{column}` → {full_form}**"
        )

        st.caption(
            meaning
        )

        st.divider()


# =========================================================
# FIND RELEVANT CSV COLUMNS
# =========================================================

def find_relevant_columns(question, df):

    question_normalized = normalize_text(
        question
    )

    found = []

    for column in df.columns:

        column_text = normalize_text(
            column
        )

        # Direct column match
        if column_text in question_normalized:

            found.append(column)
            continue

        # Synonym matching
        key = column_text.replace(
            " ",
            ""
        )

        if key in COLUMN_SYNONYMS:

            for synonym in COLUMN_SYNONYMS[key]:

                if normalize_text(
                    synonym
                ) in question_normalized:

                    found.append(column)
                    break

    return list(dict.fromkeys(found))


# =========================================================
# CSV DIRECT ANALYSIS
# =========================================================

def direct_csv_analysis(question, df):

    q = normalize_text(question)

    relevant_columns = find_relevant_columns(
        question,
        df
    )

    # -----------------------------------------
    # ROW COUNT
    # -----------------------------------------

    if any(word in q for word in [
        "how many rows",
        "number of rows",
        "total rows",
        "row count",
        "how many records",
        "number of records",
        "total records",
        "how many patients",
        "number of patients"
    ]):

        return (
            f"The dataset contains **{len(df)} rows/records**.",
            relevant_columns
        )

    # -----------------------------------------
    # COLUMN COUNT
    # -----------------------------------------

    if any(word in q for word in [
        "how many columns",
        "number of columns",
        "total columns",
        "column count"
    ]):

        return (
            f"The dataset contains **{len(df.columns)} columns**.",
            relevant_columns
        )

    # -----------------------------------------
    # COLUMN NAMES
    # -----------------------------------------

    if any(word in q for word in [
        "column names",
        "columns are",
        "what columns",
        "list columns",
        "show columns",
        "all columns"
    ]):

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
            f"The dataset shape is **{df.shape}** "
            f"(rows × columns).",
            relevant_columns
        )

    # -----------------------------------------
    # MISSING VALUES
    # -----------------------------------------

    if any(word in q for word in [
        "missing",
        "null",
        "empty values",
        "missing values"
    ]):

        total_missing = int(
            df.isnull().sum().sum()
        )

        return (
            f"The dataset contains "
            f"**{total_missing} missing values**.",
            relevant_columns
        )

    # -----------------------------------------
    # COLUMN MEANING / FULL FORM
    # -----------------------------------------

    if relevant_columns and any(word in q for word in [
        "meaning",
        "full form",
        "what does",
        "what is",
        "stands for"
    ]):

        if len(relevant_columns) == 1:

            column = relevant_columns[0]

            full_form, meaning = get_column_info(
                column
            )

            return (
                f"**{column}** means "
                f"**{full_form}**.\n\n"
                f"{meaning}",
                relevant_columns
            )

        return (
            "I found the requested columns.",
            relevant_columns
        )

    # -----------------------------------------
    # DIABETIC COUNT
    # -----------------------------------------

    if (
        "outcome" in [
            normalize_text(c)
            for c in df.columns
        ]
        and any(word in q for word in [
            "how many diabetic",
            "number of diabetic",
            "diabetic patients"
        ])
    ):

        outcome_col = None

        for c in df.columns:

            if normalize_text(c) == "outcome":

                outcome_col = c
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
                f"There are **{count} diabetic "
                f"patients** in the dataset "
                f"where Outcome = 1.",
                relevant_columns
            )

    # -----------------------------------------
    # NUMERICAL ANALYSIS
    # -----------------------------------------

    if relevant_columns:

        column = relevant_columns[0]

        numeric = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if len(numeric) == 0:

            return (
                f"The column **{column}** does not "
                f"contain usable numerical values.",
                relevant_columns
            )

        # Average / Mean
        if any(word in q for word in [
            "average",
            "avg",
            "mean",
            "avarage"
        ]):

            value = numeric.mean()

            return (
                f"The average **{column}** is "
                f"**{value:.2f}**.",
                relevant_columns
            )

        # Highest / Maximum
        if any(word in q for word in [
            "highest",
            "maximum",
            "max",
            "largest",
            "highest value"
        ]):

            value = numeric.max()

            return (
                f"The highest **{column}** value is "
                f"**{value:.2f}**.",
                relevant_columns
            )

        # Lowest / Minimum
        if any(word in q for word in [
            "lowest",
            "minimum",
            "min",
            "smallest",
            "lowest value"
        ]):

            value = numeric.min()

            return (
                f"The lowest **{column}** value is "
                f"**{value:.2f}**.",
                relevant_columns
            )

        # Sum
        if any(word in q for word in [
            "sum",
            "total"
        ]):

            value = numeric.sum()

            return (
                f"The total **{column}** is "
                f"**{value:.2f}**.",
                relevant_columns
            )

        # Median
        if "median" in q:

            value = numeric.median()

            return (
                f"The median **{column}** is "
                f"**{value:.2f}**.",
                relevant_columns
            )

    return None, relevant_columns


# =========================================================
# TXT Q&A SEARCH
# =========================================================

def search_qa_dataset(question, text):

    if not text:
        return None

    entries = []

    for line in text.splitlines():

        if "|" not in line:
            continue

        parts = line.split(
            "|",
            1
        )

        if len(parts) != 2:
            continue

        q = parts[0].strip()
        a = parts[1].strip()

        if q and a:

            entries.append(
                (q, a)
            )

    if not entries:
        return None

    normalized_question = normalize_text(
        question
    )

    # Exact match
    for stored_q, answer in entries:

        if normalize_text(
            stored_q
        ) == normalized_question:

            return answer

    # Fuzzy match
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
# LOCAL INTENT
# =========================================================

def local_intent_answer(question):

    q = normalize_text(question)

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
            "I'm IntelliMind AI, an AI-powered "
            "question-answering assistant."
        )

    if "what can you do" in q:

        return (
            "I can analyze uploaded CSV files, "
            "read TXT/PDF/Word files, answer "
            "questions, and explain dataset fields."
        )

    if q in [
        "thanks",
        "thank you",
        "thank you so much"
    ]:

        return "You're welcome! 😊"

    if q in [
        "bye",
        "goodbye"
    ]:

        return "Goodbye! 👋"

    return None


# =========================================================
# BUILT-IN GENERAL ANSWERS
# =========================================================

def built_in_answer(question):

    q = normalize_text(question)

    answers = {

        "ai": (
            "**AI** stands for **Artificial Intelligence**. "
            "It is the technology that enables computers "
            "to perform tasks that normally require human intelligence."
        ),

        "ml": (
            "**ML** stands for **Machine Learning**. "
            "It is a branch of AI where computers learn "
            "patterns from data and make predictions or decisions."
        ),

        "machine learning": (
            "**Machine Learning (ML)** is a branch of AI "
            "where computers learn patterns from data "
            "and use those patterns to make predictions or decisions."
        ),

        "dl": (
            "**DL** stands for **Deep Learning**. "
            "It uses neural networks with multiple layers "
            "to learn complex patterns from data."
        ),

        "deep learning": (
            "**Deep Learning (DL)** is a type of machine learning "
            "that uses multi-layer neural networks."
        ),

        "nlp": (
            "**NLP** stands for **Natural Language Processing**. "
            "It helps computers understand and process human language."
        ),

        "natural language processing": (
            "**Natural Language Processing (NLP)** is a field of AI "
            "that focuses on understanding and processing human language."
        ),

        "cv": (
            "**CV** stands for **Computer Vision**. "
            "It enables computers to understand images and videos."
        ),

        "computer vision": (
            "**Computer Vision (CV)** is an AI field that helps "
            "computers analyze and understand visual information."
        ),

        "python": (
            "**Python** is a high-level programming language "
            "widely used in web development, data science, "
            "machine learning, AI and automation."
        )
    }

    if q in answers:

        return answers[q]

    return None


# =========================================================
# TXT GENERIC SEARCH
# =========================================================

def split_text(text, chunk_size=700):

    words = text.split()

    chunks = []

    current = []

    size = 0

    for word in words:

        current.append(word)
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


def search_txt(question, text):

    if not text:
        return None

    chunks = split_text(text)

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

        best_index = int(
            np.argmax(scores)
        )

        if scores[best_index] >= 0.15:

            return chunks[best_index]

    except Exception:

        pass

    return None


# =========================================================
# BUILD FILE CONTEXT
# =========================================================

def build_file_context(question):

    csv_context = []
    text_context = []

    for filename, data in st.session_state.files_data.items():

        if data["type"] == "csv":

            df = data["data"]

            relevant_columns = find_relevant_columns(
                question,
                df
            )

            if relevant_columns:

                sample = df[
                    relevant_columns
                ].head(15).to_string(
                    index=False
                )

                csv_context.append(
                    f"FILE: {filename}\n"
                    f"COLUMNS: {relevant_columns}\n"
                    f"DATA:\n{sample}"
                )

        else:

            text = data["data"]

            result = search_txt(
                question,
                text
            )

            if result:

                text_context.append(
                    f"FILE: {filename}\n"
                    f"CONTENT:\n{result}"
                )

    return (
        "\n\n".join(csv_context),
        "\n\n".join(text_context)
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
    csv_context,
    text_context
):

    client = get_gemini_client()

    if client is None:

        return None

    prompt = f"""
You are IntelliMind AI.

Answer the user's question clearly and simply.

USER QUESTION:
{question}

UPLOADED CSV DATA:
{csv_context if csv_context else "No relevant CSV data found."}

UPLOADED TEXT/PDF/WORD DATA:
{text_context if text_context else "No relevant document content found."}

RULES:

1. If the question is about uploaded CSV data,
   use the provided CSV data.

2. Never invent numerical values from CSV.

3. If the answer is available in uploaded documents,
   use that information.

4. For general questions, answer normally.

5. Keep the answer simple and useful.

6. Do not mention internal processing.

7. Do not show matching rows.

8. Do not show CSV evidence.

9. Do not create a "Relevant Column" table.

10. If information is not available, say that clearly.
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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 Upload Files")

    uploaded_files = st.file_uploader(
        "Upload CSV, TXT, PDF or Word files",
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

            extension = filename.lower().split(
                "."
            )[-1]

            try:

                # =========================
                # CSV
                # =========================

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


                # =========================
                # TXT
                # =========================

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


                # =========================
                # PDF
                # =========================

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


                # =========================
                # WORD
                # =========================

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

                st.success(
                    f"Loaded: {filename}"
                )

            except Exception as e:

                st.error(
                    f"Could not read {filename}: {e}"
                )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()

    st.divider()

    st.caption(
        "Supported: CSV • TXT • PDF • DOCX"
    )


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.caption(
    "AI-powered file analysis and question answering"
)


# =========================================================
# FILE OVERVIEW
# =========================================================

if st.session_state.files_data:

    st.subheader("📂 Uploaded Files")

    for filename, data in st.session_state.files_data.items():

        file_type = data["type"]

        # ---------------- CSV ----------------

        if file_type == "csv":

            df = data["data"]

            with st.expander(
                f"📊 {filename} — CSV Dataset"
            ):

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Rows",
                    len(df)
                )

                col2.metric(
                    "Columns",
                    len(df.columns)
                )

                col3.metric(
                    "Missing Values",
                    int(
                        df.isnull().sum().sum()
                    )
                )

                st.markdown(
                    "### 📌 Dataset Columns"
                )

                show_column_information(
                    list(df.columns)
                )

                st.markdown(
                    "### 👀 Preview"
                )

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )


        # ---------------- TEXT / PDF / WORD ----------------

        else:

            text = data["data"]

            with st.expander(
                f"📄 {filename} — {file_type.upper()} Document"
            ):

                st.metric(
                    "Characters",
                    len(text)
                )

                st.metric(
                    "Lines",
                    len(text.splitlines())
                )

                # PDF / Word headings
                possible_fields = (
                    extract_possible_fields(text)
                )

                table_columns = (
                    extract_table_columns_from_text(
                        text
                    )
                )

                detected = list(
                    dict.fromkeys(
                        table_columns +
                        possible_fields
                    )
                )

                if detected:

                    st.markdown(
                        "### 📚 Detected Fields / Headings"
                    )

                    show_column_information(
                        detected
                    )

                st.markdown(
                    "### 📖 Document Preview"
                )

                preview = text[:5000]

                st.text_area(
                    "Content",
                    preview,
                    height=250
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

        # Show column information only after answer
        if (
            message["role"] == "assistant"
            and message.get("columns")
        ):

            show_column_information(
                message["columns"]
            )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your uploaded files..."
)


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

    with st.chat_message("user"):

        st.markdown(question)

    answer = None
    answer_columns = []

    # =====================================================
    # 1. CSV DIRECT ANALYSIS
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
    # 2. TXT / PDF / DOCX Q&A SEARCH
    # =====================================================

    if answer is None:

        for filename, data in st.session_state.files_data.items():

            if data["type"] in [
                "txt",
                "pdf",
                "docx"
            ]:

                text = data["data"]

                result = search_qa_dataset(
                    question,
                    text
                )

                if result:

                    answer = result

                    break


    # =====================================================
    # 3. LOCAL INTENT
    # =====================================================

    if answer is None:

        answer = local_intent_answer(
            question
        )


    # =====================================================
    # 4. BUILT-IN GENERAL ANSWER
    # =====================================================

    if answer is None:

        answer = built_in_answer(
            question
        )


    # =====================================================
    # 5. GEMINI
    # =====================================================

    if answer is None:

        csv_context, text_context = (
            build_file_context(question)
        )

        answer = generate_ai_answer(
            question,
            csv_context,
            text_context
        )


    # =====================================================
    # 6. FALLBACK
    # =====================================================

    if answer is None:

        answer = (
            "Sorry, I couldn't find a reliable answer "
            "right now. Please try asking the question "
            "in a different way."
        )


    # =====================================================
    # SHOW ANSWER
    # =====================================================

    with st.chat_message("assistant"):

        st.markdown(answer)

        # Only show column information
        # when a CSV column was actually used.
        if answer_columns:

            show_column_information(
                answer_columns
            )


    # =====================================================
    # SAVE HISTORY
    # =====================================================

    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer,
            "columns": answer_columns
        }
    )
