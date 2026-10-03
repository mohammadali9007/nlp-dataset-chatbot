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


# =========================================================
# SESSION STATE
# =========================================================

if "files_data" not in st.session_state:
    st.session_state.files_data = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = ""

if "active_file" not in st.session_state:
    st.session_state.active_file = None

if "response_language" not in st.session_state:
    st.session_state.response_language = "English"


# =========================================================
# COLUMN DICTIONARY
# =========================================================

COLUMN_INFO = {

    # General
    "id": ("ID", "Unique identification number."),
    "name": ("Name", "Name of the person or item."),
    "age": ("Age", "Age of a person."),
    "gender": ("Gender", "Gender of a person."),
    "dob": ("Date of Birth", "Birth date."),
    "email": ("Email", "Email address."),
    "phone": ("Phone Number", "Contact phone number."),
    "address": ("Address", "Residential or business address."),
    "status": ("Status", "Current state or condition."),

    # Academic
    "gpa": ("Grade Point Average", "Academic grade average."),
    "cgpa": ("Cumulative Grade Point Average", "Overall academic grade average."),
    "studentid": ("Student ID", "Unique student identification."),
    "roll": ("Roll Number", "Student roll number."),
    "department": ("Department", "Academic or organizational department."),
    "semester": ("Semester", "Academic semester."),

    # AI / Computer Science
    "ai": ("Artificial Intelligence", "Technology that enables machines to perform intelligent tasks."),
    "ml": ("Machine Learning", "A method where computers learn patterns from data."),
    "dl": ("Deep Learning", "Machine learning using multi-layer neural networks."),
    "nlp": ("Natural Language Processing", "AI techniques for processing human language."),
    "cv": ("Computer Vision", "AI techniques for understanding images and videos."),

    # Finance
    "salary": ("Salary", "Amount of money received as regular payment."),
    "income": ("Income", "Money received from work, business, or other sources."),
    "annualincome": ("Annual Income", "Total income received in one year."),
    "price": ("Price", "Cost or monetary value of an item."),
    "quantity": ("Quantity", "Number or amount of items."),
    "credit_score": ("Credit Score", "Numerical measure of creditworthiness."),
    "loan_status": ("Loan Status", "Current status of a loan."),

    # Employment
    "experience": ("Work Experience", "Amount of professional experience."),
    "performancescore": ("Performance Score", "Numerical measurement of performance."),
    "performance_score": ("Performance Score", "Numerical measurement of performance."),

    # Health
    "bmi": ("Body Mass Index", "Measure calculated using height and weight."),
    "glucose": ("Blood Glucose", "Amount of glucose in the blood."),
    "bloodpressure": ("Blood Pressure", "Pressure of blood against blood vessel walls."),
    "bp": ("Blood Pressure", "Blood pressure measurement."),
    "pregnancies": ("Number of Pregnancies", "Number of pregnancies."),
    "insulin": ("Insulin", "Hormone involved in controlling blood glucose."),
    "outcome": ("Outcome", "Result or target classification."),
    "weight": ("Weight", "Body or object weight."),
    "height": ("Height", "Height measurement."),

    # Kidney dataset
    "sc": ("Serum Creatinine", "Blood creatinine level."),
    "bu": ("Blood Urea", "Amount of urea in the blood."),
    "hemo": ("Hemoglobin", "Protein in red blood cells that carries oxygen."),
    "pcv": ("Packed Cell Volume", "Percentage of blood volume occupied by red blood cells."),
    "sod": ("Sodium", "Sodium level in the blood."),
    "pot": ("Potassium", "Potassium level in the blood."),
    "htn": ("Hypertension", "High blood pressure condition."),
    "dm": ("Diabetes Mellitus", "Diabetes condition."),
    "cad": ("Coronary Artery Disease", "Disease affecting the coronary arteries."),
    "classification": ("Classification", "Category or predicted class."),
    "rbc": ("Red Blood Cells", "Red blood cell measurement or category."),
    "pc": ("Pus Cells", "Pus cell measurement or category."),
    "pcc": ("Pus Cell Clumps", "Presence of pus cell clumps."),
    "ba": ("Bacteria", "Presence or level of bacteria."),
    "bgr": ("Blood Glucose Random", "Random blood glucose measurement."),
    "sg": ("Specific Gravity", "Measure of urine concentration."),
    "al": ("Albumin", "Albumin level."),
    "su": ("Sugar", "Urine sugar measurement."),
    "pe": ("Pedal Edema", "Swelling of the feet or legs."),
    "ane": ("Anemia", "Condition related to low healthy red blood cells or hemoglobin."),
    "appet": ("Appetite", "Appetite condition.")
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def clean_column_name(column):
    value = str(column).strip()

    # camelCase
    value = re.sub(r"([a-z])([A-Z])", r"\1 \2", value)

    # underscore / hyphen
    value = value.replace("_", " ")
    value = value.replace("-", " ")

    value = re.sub(r"\s+", " ", value).strip()

    return value


def column_key(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def get_column_info(column):
    key = column_key(column)

    if key in COLUMN_INFO:
        return COLUMN_INFO[key]

    readable = clean_column_name(column).title()

    return (
        readable,
        f"Auto-readable name for {readable}."
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
        return pd.read_csv(uploaded_file)

    except Exception:
        try:
            uploaded_file.seek(0)
            return pd.read_csv(uploaded_file, encoding="latin1")

        except Exception:
            try:
                uploaded_file.seek(0)
                return pd.read_csv(uploaded_file, encoding="cp1252")

            except Exception:
                return None


def read_txt_file(uploaded_file):

    raw = uploaded_file.read()

    for encoding in ["utf-8", "utf-8-sig", "cp1252", "latin1"]:

        try:
            return raw.decode(encoding)
        except Exception:
            continue

    return raw.decode("utf-8", errors="ignore")


def read_pdf_file(uploaded_file):

    try:

        uploaded_file.seek(0)

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

    except Exception:
        return ""


def read_docx_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        document = Document(uploaded_file)

        parts = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                parts.append(text)

        # Read tables
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
# DOCUMENT FIELD DETECTION
# =========================================================

def detect_document_fields(text):

    fields = []

    if not text:
        return fields

    lines = text.splitlines()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # field: value
        if ":" in line:

            left = line.split(":", 1)[0].strip()

            if (
                1 <= len(left.split()) <= 6
                and len(left) <= 50
            ):
                fields.append(left)

        # table header
        elif "|" in line:

            parts = [
                x.strip()
                for x in line.split("|")
                if x.strip()
            ]

            if 2 <= len(parts) <= 15:

                for part in parts:

                    if len(part) <= 40:
                        fields.append(part)

        # Heading
        elif line.startswith("#"):

            heading = line.lstrip("#").strip()

            if heading:
                fields.append(heading)

    # Remove duplicates
    unique = []

    for field in fields:

        if field not in unique:
            unique.append(field)

    return unique[:30]


# =========================================================
# CSV PROFILE
# =========================================================

def get_csv_profile(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
    ]

    missing_total = int(
        df.isna().sum().sum()
    )

    duplicate_rows = int(
        df.duplicated().sum()
    )

    return {
        "rows": len(df),
        "columns": len(df.columns),
        "numeric": numeric_columns,
        "categorical": categorical_columns,
        "missing": missing_total,
        "duplicates": duplicate_rows
    }


# =========================================================
# DYNAMIC QUESTION GENERATOR
# =========================================================

def generate_dataset_questions(df):

    questions = {
        "Overview": [],
        "Statistics": [],
        "Insights": [],
        "Data Quality": [],
        "Columns": []
    }

    if df is None or df.empty:
        return questions

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    questions["Overview"].append(
        "How many rows are in this dataset?"
    )

    questions["Overview"].append(
        "How many columns are in this dataset?"
    )

    questions["Overview"].append(
        "What are the column names?"
    )

    questions["Overview"].append(
        "Give me an overview of this dataset."
    )

    # -----------------------------------------------------
    # STATISTICS
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_columns[:6]:

        questions["Statistics"].append(
            f"What is the average {clean_column_name(col)}?"
        )

        questions["Statistics"].append(
            f"What is the highest {clean_column_name(col)}?"
        )

        questions["Statistics"].append(
            f"What is the lowest {clean_column_name(col)}?"
        )

    # -----------------------------------------------------
    # INSIGHTS
    # -----------------------------------------------------

    if numeric_columns:

        questions["Insights"].append(
            f"Which column has the highest average value?"
        )

        questions["Insights"].append(
            f"Give me the main statistical insights."
        )

    categorical_columns = [
        col for col in df.columns
        if col not in numeric_columns
    ]

    for col in categorical_columns[:3]:

        questions["Insights"].append(
            f"What are the most common values in {clean_column_name(col)}?"
        )

    # -----------------------------------------------------
    # DATA QUALITY
    # -----------------------------------------------------

    questions["Data Quality"].append(
        "Are there any missing values?"
    )

    questions["Data Quality"].append(
        "How many duplicate rows are there?"
    )

    questions["Data Quality"].append(
        "Which columns have missing values?"
    )

    questions["Data Quality"].append(
        "Show me the data quality summary."
    )

    # -----------------------------------------------------
    # COLUMNS
    # -----------------------------------------------------

    for col in df.columns[:8]:

        readable, _ = get_column_info(col)

        questions["Columns"].append(
            f"What does {col} mean?"
        )

        questions["Columns"].append(
            f"What is the full form of {col}?"
        )

    # Remove duplicates
    for category in questions:

        unique_questions = []

        for q in questions[category]:

            if q not in unique_questions:
                unique_questions.append(q)

        questions[category] = unique_questions[:10]

    return questions


# =========================================================
# DOCUMENT QUESTIONS
# =========================================================

def generate_document_questions(text):

    questions = {
        "Overview": [
            "Give me a summary of this document.",
            "What is this document mainly about?",
            "What are the main points?"
        ],
        "Insights": [
            "What are the most important details?",
            "Extract the key information."
        ],
        "Columns": [],
        "Statistics": [],
        "Data Quality": []
    }

    fields = detect_document_fields(text)

    for field in fields[:6]:

        questions["Columns"].append(
            f"What does {field} mean?"
        )

    return questions


# =========================================================
# DIRECT CSV ANSWER ENGINE
# =========================================================

def direct_csv_analysis(question, df):

    if df is None or df.empty:
        return None

    q = normalize_text(question)

    profile = get_csv_profile(df)

    # -----------------------------------------------------
    # ROW COUNT
    # -----------------------------------------------------

    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
        or "how many records" in q
    ):

        return (
            f"This dataset contains **{profile['rows']:,} rows**."
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
            f"This dataset contains **{profile['columns']} columns**."
        )

    # -----------------------------------------------------
    # COLUMN NAMES
    # -----------------------------------------------------

    if (
        "column names" in q
        or "columns are" in q
        or "what columns" in q
    ):

        names = ", ".join(
            str(x) for x in df.columns
        )

        return f"The columns are:\n\n**{names}**"

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    if (
        "overview" in q
        or "summarize dataset" in q
        or "summary of dataset" in q
    ):

        return (
            f"### Dataset Overview\n\n"
            f"- **Rows:** {profile['rows']:,}\n"
            f"- **Columns:** {profile['columns']}\n"
            f"- **Numeric fields:** {len(profile['numeric'])}\n"
            f"- **Text/category fields:** {len(profile['categorical'])}\n"
            f"- **Missing values:** {profile['missing']:,}\n"
            f"- **Duplicate rows:** {profile['duplicates']:,}"
        )

    # -----------------------------------------------------
    # MISSING VALUES
    # -----------------------------------------------------

    if (
        "missing" in q
        or "null" in q
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
                "There are **no missing values** in this dataset."
            )

        lines = []

        for col, value in missing.items():

            lines.append(
                f"- **{col}:** {int(value):,}"
            )

        return (
            "### Missing Values\n\n"
            + "\n".join(lines)
        )

    # -----------------------------------------------------
    # DUPLICATES
    # -----------------------------------------------------

    if "duplicate" in q:

        return (
            f"This dataset contains "
            f"**{profile['duplicates']:,} duplicate rows**."
        )

    # -----------------------------------------------------
    # COLUMN MEANING / FULL FORM
    # -----------------------------------------------------

    for col in df.columns:

        col_normal = normalize_text(col)
        readable, meaning = get_column_info(col)

        if (
            col_normal in q
            or clean_column_name(col).lower() in q
        ):

            if (
                "full form" in q
                or "meaning" in q
                or "what does" in q
            ):

                return (
                    f"**{col}** → **{readable}**\n\n"
                    f"{meaning}"
                )

    # -----------------------------------------------------
    # NUMERIC COLUMN ANALYSIS
    # -----------------------------------------------------

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    for col in numeric_columns:

        col_name = clean_column_name(col)

        col_normal = normalize_text(col)

        matched = (
            col_normal in q
            or col_name.lower() in q
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
                f"The average **{col_name}** is "
                f"**{series.mean():,.2f}**."
            )

        # Highest
        if (
            "highest" in q
            or "maximum" in q
            or "max" in q
        ):

            return (
                f"The highest **{col_name}** is "
                f"**{series.max():,.2f}**."
            )

        # Lowest
        if (
            "lowest" in q
            or "minimum" in q
            or "min" in q
        ):

            return (
                f"The lowest **{col_name}** is "
                f"**{series.min():,.2f}**."
            )

        # Median
        if "median" in q:

            return (
                f"The median **{col_name}** is "
                f"**{series.median():,.2f}**."
            )

        # Sum
        if (
            "total" in q
            or "sum" in q
        ):

            return (
                f"The total **{col_name}** is "
                f"**{series.sum():,.2f}**."
            )

    # -----------------------------------------------------
    # CATEGORICAL COLUMN
    # -----------------------------------------------------

    for col in profile["categorical"]:

        col_normal = normalize_text(col)

        if col_normal not in q:
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

            result = []

            for value, count in counts.items():

                result.append(
                    f"- **{value}**: {int(count):,}"
                )

            return (
                f"### Most Common Values in {col}\n\n"
                + "\n".join(result)
            )

        if (
            "unique" in q
            or "different values" in q
        ):

            count = df[col].nunique(
                dropna=True
            )

            return (
                f"**{col}** contains "
                f"**{count:,} unique values**."
            )

    return None


# =========================================================
# GLOBAL DATASET INSIGHTS
# =========================================================

def generate_quick_insights(df):

    if df is None or df.empty:
        return []

    insights = []

    profile = get_csv_profile(df)

    insights.append(
        f"📊 Dataset has **{profile['rows']:,} records** "
        f"and **{profile['columns']} fields**."
    )

    if profile["numeric"]:

        insights.append(
            f"🔢 **{len(profile['numeric'])} numeric fields** "
            f"are available for statistical analysis."
        )

    if profile["categorical"]:

        insights.append(
            f"🏷️ **{len(profile['categorical'])} text/category fields** "
            f"are available."
        )

    if profile["missing"] > 0:

        missing = df.isna().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if not missing.empty:

            top_missing = missing.index[0]

            insights.append(
                f"⚠️ **{top_missing}** has the highest number "
                f"of missing values."
            )

    else:

        insights.append(
            "✅ No missing values were detected."
        )

    if profile["duplicates"] > 0:

        insights.append(
            f"♻️ There are **{profile['duplicates']:,} duplicate rows**."
        )

    else:

        insights.append(
            "✨ No duplicate rows were detected."
        )

    return insights[:5]


# =========================================================
# TXT Q&A SEARCH
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
                (question, answer)
            )

    return pairs


def search_qa_dataset(question, text):

    pairs = parse_qa_text(text)

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
# GENERIC TEXT SEARCH
# =========================================================

def split_text(text, chunk_size=700):

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


def search_text(question, text):

    chunks = split_text(text)

    if not chunks:
        return None

    if len(chunks) == 1:
        return chunks[0]

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        matrix = vectorizer.fit_transform(
            chunks
        )

        question_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            question_vector,
            matrix
        )[0]

        top_indices = np.argsort(scores)[::-1][:3]

        selected = []

        for idx in top_indices:

            if scores[idx] >= 0.10:
                selected.append(
                    chunks[idx]
                )

        if selected:
            return "\n\n".join(selected)

    except Exception:
        pass

    return None


# =========================================================
# BUILT-IN ANSWERS
# =========================================================

BUILT_IN = {

    "python": (
        "Python is a high-level programming language "
        "used for web development, automation, data science, "
        "machine learning and AI."
    ),

    "machine learning": (
        "Machine Learning is a branch of AI where computers "
        "learn patterns from data and use those patterns "
        "to make predictions or decisions."
    ),

    "ml": (
        "Machine Learning is a branch of AI where computers "
        "learn patterns from data and use those patterns "
        "to make predictions or decisions."
    ),

    "deep learning": (
        "Deep Learning is a type of Machine Learning that "
        "uses multi-layer neural networks."
    ),

    "dl": (
        "Deep Learning is a type of Machine Learning that "
        "uses multi-layer neural networks."
    ),

    "nlp": (
        "NLP stands for Natural Language Processing. "
        "It allows computers to process and understand human language."
    ),

    "natural language processing": (
        "Natural Language Processing is an AI field that "
        "focuses on understanding and processing human language."
    ),

    "computer vision": (
        "Computer Vision is an AI field that enables computers "
        "to understand images and videos."
    ),

    "cv": (
        "Computer Vision is an AI field that enables computers "
        "to understand images and videos."
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


def built_in_answer(question):

    q = normalize_text(question)

    # Exact
    if q in BUILT_IN:
        return BUILT_IN[q]

    # Contains
    for key, answer in BUILT_IN.items():

        if key in q:
            return answer

    return None


# =========================================================
# GEMINI
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


def generate_ai_answer(
    question,
    context,
    chat_history=None
):

    client = get_gemini_client()

    if client is None:
        return None

    language = st.session_state.response_language

    if language == "Bangla":
        language_instruction = (
            "Answer in simple Bangla. "
            "You may use English technical terms when necessary."
        )

    elif language == "Auto":
        language_instruction = (
            "Answer in the same language used by the user."
        )

    else:
        language_instruction = (
            "Answer in simple and clear English."
        )

    history_text = ""

    if chat_history:

        recent = chat_history[-6:]

        for item in recent:

            history_text += (
                f"User: {item['question']}\n"
                f"Assistant: {item['answer']}\n\n"
            )

    prompt = f"""
You are IntelliMind AI, a reliable data copilot.

Rules:

1. Answer the user's question directly.
2. If CSV data is provided, use ONLY the supplied data for numerical claims.
3. Never invent CSV values.
4. If the answer is not available in the supplied context, clearly say so.
5. Do not create fake statistics.
6. Keep the answer concise but useful.
7. Use simple language.
8. {language_instruction}

Previous conversation:
{history_text}

Available context:
{context}

User question:
{question}
"""

    try:

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt
        )

        if response and response.text:
            return response.text.strip()

    except Exception:
        return None

    return None


# =========================================================
# FILE CONTEXT
# =========================================================

def create_file_context():

    contexts = []

    for filename, data in st.session_state.files_data.items():

        file_type = data.get(
            "type",
            ""
        )

        if file_type == "csv":

            df = data["data"]

            contexts.append(
                f"""
FILE: {filename}

CSV shape:
{df.shape}

Columns:
{list(df.columns)}

Data types:
{df.dtypes.to_string()}

Sample:
{df.head(8).to_string(index=False)}
"""
            )

        else:

            text = data.get(
                "text",
                ""
            )

            contexts.append(
                f"""
FILE: {filename}

CONTENT:
{text[:12000]}
"""
            )

    return "\n\n".join(contexts)


# =========================================================
# FIND RELEVANT COLUMN
# =========================================================

def find_relevant_columns(
    question,
    df
):

    if df is None:
        return []

    q = normalize_text(question)

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

        for name in names:

            if name and name in q:

                matches.append(
                    original
                )

                break

    return list(
        dict.fromkeys(matches)
    )


# =========================================================
# DISPLAY COLUMN INFO
# =========================================================

def show_relevant_column_info(
    question,
    df
):

    matches = find_relevant_columns(
        question,
        df
    )

    if not matches:
        return

    st.caption("📌 Field")

    for col in matches[:3]:

        readable, _ = get_column_info(
            col
        )

        st.write(
            f"`{col}` → **{readable}**"
        )


# =========================================================
# PROCESS QUESTION
# =========================================================

def process_question(question):

    question = question.strip()

    if not question:
        return

    # Add user message later after answer
    answer = None
    relevant_df = None
    relevant_question = question

    # -----------------------------------------------------
    # 1. CSV DIRECT ANALYSIS
    # -----------------------------------------------------

    active_file = st.session_state.get(
        "active_file"
    )

    if active_file:

        file_data = st.session_state.files_data.get(
            active_file
        )

        if (
            file_data
            and file_data.get("type") == "csv"
        ):

            relevant_df = file_data["data"]

            answer = direct_csv_analysis(
                question,
                relevant_df
            )

    # If no direct answer, check all CSV files
    if answer is None:

        for filename, data in st.session_state.files_data.items():

            if data.get("type") != "csv":
                continue

            result = direct_csv_analysis(
                question,
                data["data"]
            )

            if result:

                answer = result
                relevant_df = data["data"]

                st.session_state.active_file = filename

                break

    # -----------------------------------------------------
    # 2. TEXT / DOCUMENT SEARCH
    # -----------------------------------------------------

    if answer is None:

        for filename, data in st.session_state.files_data.items():

            if data.get("type") == "csv":
                continue

            text = data.get(
                "text",
                ""
            )

            if not text:
                continue

            # TXT Q&A
            result = search_qa_dataset(
                question,
                text
            )

            if result:

                answer = result
                break

            # Generic document search
            result = search_text(
                question,
                text
            )

            if result:

                answer = result
                break

    # -----------------------------------------------------
    # 3. BUILT-IN ANSWERS
    # -----------------------------------------------------

    if answer is None:

        answer = built_in_answer(
            question
        )

    # -----------------------------------------------------
    # 4. GEMINI
    # -----------------------------------------------------

    if answer is None:

        context = create_file_context()

        answer = generate_ai_answer(
            question,
            context,
            st.session_state.chat_history
        )

    # -----------------------------------------------------
    # 5. FALLBACK
    # -----------------------------------------------------

    if answer is None:

        answer = (
            "I couldn't find a reliable answer from the "
            "uploaded files or local knowledge. "
            "Please provide more context or ask the question differently."
        )

    # -----------------------------------------------------
    # SAVE HISTORY
    # -----------------------------------------------------

    st.session_state.chat_history.append({
        "question": question,
        "answer": answer
    })

    # -----------------------------------------------------
    # DISPLAY
    # -----------------------------------------------------

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        st.markdown(answer)

        if relevant_df is not None:

            show_relevant_column_info(
                question,
                relevant_df
            )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("🤖 IntelliMind AI")

    st.caption(
        "Your Personal Data Copilot"
    )

    st.divider()

    # -----------------------------------------------------
    # FILE UPLOAD
    # -----------------------------------------------------

    st.subheader("📁 Files")

    uploaded_files = st.file_uploader(
        "Upload your files",
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

            extension = filename.lower().split(".")[-1]

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

                    if st.session_state.active_file is None:
                        st.session_state.active_file = filename

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
    # ACTIVE FILE
    # -----------------------------------------------------

    csv_files = [
        filename
        for filename, data
        in st.session_state.files_data.items()
        if data.get("type") == "csv"
    ]

    if csv_files:

        st.subheader("🎯 Active Dataset")

        current = (
            st.session_state.active_file
            if st.session_state.active_file in csv_files
            else csv_files[0]
        )

        selected_file = st.selectbox(
            "Choose dataset",
            csv_files,
            index=csv_files.index(current)
        )

        st.session_state.active_file = selected_file

    # -----------------------------------------------------
    # FIELD DICTIONARY
    # -----------------------------------------------------

    st.divider()

    st.subheader("📚 Fields")

    active_df = None

    if st.session_state.active_file:

        active_data = st.session_state.files_data.get(
            st.session_state.active_file
        )

        if (
            active_data
            and active_data.get("type") == "csv"
        ):

            active_df = active_data["data"]

    if active_df is not None:

        # ONLY SHORT LIST IN SIDEBAR
        columns = list(
            active_df.columns
        )

        for col in columns[:8]:

            readable, _ = get_column_info(
                col
            )

            st.caption(
                f"`{col}` → {readable}"
            )

        # Detailed dictionary hidden
        if len(columns) > 8:

            st.caption(
                f"+ {len(columns) - 8} more fields"
            )

        with st.expander(
            "🔎 View all fields"
        ):

            for col in columns:

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

        # Document fields
        document_fields = []

        for filename, data in st.session_state.files_data.items():

            if data.get("type") != "csv":

                fields = detect_document_fields(
                    data.get("text", "")
                )

                document_fields.extend(
                    fields
                )

        document_fields = list(
            dict.fromkeys(document_fields)
        )

        if document_fields:

            for field in document_fields[:8]:

                readable, _ = get_column_info(
                    field
                )

                st.caption(
                    f"`{field}` → {readable}"
                )

            if len(document_fields) > 8:

                with st.expander(
                    "🔎 View all fields"
                ):

                    for field in document_fields:

                        readable, meaning = get_column_info(
                            field
                        )

                        st.markdown(
                            f"**`{field}`** → {readable}"
                        )

                        st.caption(
                            meaning
                        )

        else:

            st.caption(
                "Upload a file to see fields."
            )

    # -----------------------------------------------------
    # SETTINGS
    # -----------------------------------------------------

    st.divider()

    st.subheader("⚙️ Settings")

    st.session_state.response_language = st.selectbox(
        "Response language",
        [
            "English",
            "Bangla",
            "Auto"
        ],
        index=[
            "English",
            "Bangla",
            "Auto"
        ].index(
            st.session_state.response_language
        )
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

        st.session_state.active_file = None

        st.rerun()


# =========================================================
# MAIN HEADER
# =========================================================

st.title("🤖 IntelliMind AI")

st.caption(
    "Your Personal Data Copilot • "
    "Ask questions, understand files, discover insights."
)


# =========================================================
# FILE STATUS
# =========================================================

if not st.session_state.files_data:

    st.info(
        "👋 Upload a CSV, TXT, PDF or DOCX file from the sidebar to get started."
    )

else:

    # =====================================================
    # CSV OVERVIEW
    # =====================================================

    if active_df is not None:

        profile = get_csv_profile(
            active_df
        )

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Records",
                f"{profile['rows']:,}"
            )

        with c2:
            st.metric(
                "Fields",
                profile["columns"]
            )

        with c3:
            st.metric(
                "Numeric",
                len(profile["numeric"])
            )

        with c4:
            st.metric(
                "Missing",
                f"{profile['missing']:,}"
            )

        # =================================================
        # QUICK INSIGHTS
        # =================================================

        st.subheader("⚡ Quick Insights")

        insights = generate_quick_insights(
            active_df
        )

        for insight in insights:

            st.write(
                insight
            )

        # =================================================
        # DATA PREVIEW
        # =================================================

        with st.expander(
            "👀 Preview Dataset"
        ):

            st.dataframe(
                active_df.head(10),
                use_container_width=True
            )

        # =================================================
        # SUGGESTIONS
        # =================================================

        st.divider()

        st.subheader(
            "💡 What would you like to know?"
        )

        questions = generate_dataset_questions(
            active_df
        )

    else:

        # Document mode
        st.subheader(
            "💡 What would you like to know?"
        )

        all_text = ""

        for filename, data in st.session_state.files_data.items():

            if data.get("type") != "csv":

                all_text += "\n" + data.get(
                    "text",
                    ""
                )

        questions = generate_document_questions(
            all_text
        )


    # =====================================================
    # QUESTION CATEGORIES
    # =====================================================

    category_tabs = st.tabs([
        "📊 Overview",
        "🔢 Statistics",
        "💡 Insights",
        "🛡️ Data Quality",
        "📚 Columns"
    ])

    category_map = [
        "Overview",
        "Statistics",
        "Insights",
        "Data Quality",
        "Columns"
    ]

    for tab, category in zip(
        category_tabs,
        category_map
    ):

        with tab:

            category_questions = questions.get(
                category,
                []
            )

            if not category_questions:

                st.caption(
                    "No suggestions available for this category."
                )

            else:

                # Show maximum 8
                display_questions = category_questions[:8]

                for i in range(
                    0,
                    len(display_questions),
                    2
                ):

                    cols = st.columns(2)

                    for j in range(2):

                        index = i + j

                        if index >= len(
                            display_questions
                        ):
                            continue

                        q = display_questions[index]

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

    with st.chat_message("user"):
        st.write(
            item["question"]
        )

    with st.chat_message("assistant"):

        st.markdown(
            item["answer"]
        )


# =========================================================
# SELECTED QUESTION
# =========================================================

if st.session_state.selected_question:

    selected = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = ""

    process_question(
        selected
    )


# =========================================================
# CHAT INPUT
# =========================================================

user_question = st.chat_input(
    "Ask anything about your files..."
)

if user_question:

    process_question(
        user_question
    )
