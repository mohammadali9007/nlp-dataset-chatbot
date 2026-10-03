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

if "last_uploaded_names" not in st.session_state:
    st.session_state.last_uploaded_names = []


# =========================================================
# CSS
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
        "meaning": "Blood pressure"
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
# HELPERS
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


# =========================================================
# FILE READERS
# =========================================================

def read_csv_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(
            uploaded_file
        )

        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

        return df

    except Exception:

        try:

            uploaded_file.seek(0)

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

            df.columns = [
                str(c).strip()
                for c in df.columns
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

        reader = PdfReader(
            uploaded_file
        )

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

        document = Document(
            uploaded_file
        )

        for paragraph in document.paragraphs:

            if paragraph.text.strip():

                text += (
                    paragraph.text
                    + "\n"
                )

    except Exception:
        pass

    return text


# =========================================================
# PROCESS FILES
# =========================================================

def process_uploaded_files(uploaded_files):

    processed = []

    for file in uploaded_files:

        if file is None:
            continue

        name = getattr(
            file,
            "name",
            "Unknown file"
        )

        extension = os.path.splitext(
            name
        )[1].lower()

        item = {
            "name": name,
            "type": extension,
            "df": None,
            "text": "",
            "size": getattr(
                file,
                "size",
                0
            )
        }

        if extension == ".csv":

            item["df"] = read_csv_file(
                file
            )

        elif extension == ".txt":

            item["text"] = read_txt_file(
                file
            )

        elif extension == ".pdf":

            item["text"] = read_pdf_file(
                file
            )

        elif extension == ".docx":

            item["text"] = read_docx_file(
                file
            )

        else:
            continue

        processed.append(item)

    return processed


# =========================================================
# EXACT COLUMN MATCHING
# =========================================================

def find_relevant_columns(
    question,
    df
):

    if df is None or df.empty:
        return []

    q = normalize_text(
        question
    )

    q_no_space = q.replace(
        " ",
        ""
    )

    q_words = set(
        re.findall(
            r"[a-zA-Z0-9]+",
            q
        )
    )

    exact_matches = []

    for col in df.columns:

        key = column_key(col)

        info = get_column_info(
            col
        )

        candidates = [
            normalize_text(col),
            normalize_text(info["name"]),
            key
        ]

        candidates.extend(
            normalize_text(x)
            for x in COLUMN_ALIASES.get(
                key,
                []
            )
        )

        found = False

        for candidate in candidates:

            if not candidate:
                continue

            candidate_no_space = (
                candidate.replace(
                    " ",
                    ""
                )
            )

            if len(candidate_no_space) <= 5:

                if candidate in q_words:

                    found = True
                    break

            else:

                if candidate in q:

                    found = True
                    break

                if (
                    candidate_no_space
                    in q_no_space
                ):

                    found = True
                    break

        if found:

            exact_matches.append(
                col
            )

    if exact_matches:

        return exact_matches

    # -----------------------------------------------------
    # SAFE FUZZY MATCH
    # -----------------------------------------------------

    cleaned = q

    operation_words = [
        "average",
        "avg",
        "mean",
        "maximum",
        "max",
        "highest",
        "largest",
        "minimum",
        "min",
        "lowest",
        "smallest",
        "median",
        "mode",
        "sum",
        "total",
        "count",
        "number",
        "show",
        "display",
        "give",
        "tell",
        "what is",
        "what are",
        "all",
        "every",
        "value",
        "values",
        "row",
        "rows",
        "column",
        "columns",
        "field"
    ]

    for word in operation_words:

        cleaned = re.sub(
            rf"\b{re.escape(word)}\b",
            " ",
            cleaned
        )

    cleaned = re.sub(
        r"\s+",
        " ",
        cleaned
    ).strip()

    if not cleaned:
        return []

    matches = []

    for col in df.columns:

        key = column_key(col)

        # Never fuzzy match tiny fields
        if len(key) < 4:
            continue

        info = get_column_info(
            col
        )

        names = [
            normalize_text(col),
            normalize_text(info["name"])
        ]

        names.extend(
            normalize_text(x)
            for x in COLUMN_ALIASES.get(
                key,
                []
            )
        )

        best_score = 0

        for name in names:

            score = difflib.SequenceMatcher(
                None,
                cleaned,
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
        x[0]
        for x in matches[:2]
    ]


# =========================================================
# FILE NAME MATCHING
# =========================================================

def find_requested_file(
    question,
    files_data
):

    q = normalize_text(
        question
    )

    q_no_ext = q.replace(
        ".csv",
        ""
    ).replace(
        ".txt",
        ""
    ).replace(
        ".pdf",
        ""
    ).replace(
        ".docx",
        ""
    )

    best_file = None
    best_score = 0

    for item in files_data:

        if not isinstance(item, dict):
            continue

        name = normalize_text(
            item.get(
                "name",
                ""
            )
        )

        name_no_ext = os.path.splitext(
            name
        )[0]

        name_no_ext = normalize_text(
            name_no_ext
        )

        if (
            name in q
            or name_no_ext in q_no_ext
        ):

            return item

        score = difflib.SequenceMatcher(
            None,
            name_no_ext,
            q_no_ext
        ).ratio()

        if score > best_score:

            best_score = score
            best_file = item

    if best_score >= 0.60:

        return best_file

    return None


# =========================================================
# DATA EXPLORER
# =========================================================

def explore_dataset(
    question,
    files_data
):

    q = normalize_text(
        question
    )

    # -----------------------------------------------------
    # Detect requested file
    # -----------------------------------------------------

    requested_file = find_requested_file(
        question,
        files_data
    )

    # -----------------------------------------------------
    # SELECT CSV DATASETS
    # -----------------------------------------------------

    csv_items = [
        item
        for item in files_data
        if isinstance(item, dict)
        and item.get("df") is not None
    ]

    if requested_file is not None:

        if requested_file.get(
            "df"
        ) is not None:

            csv_items = [
                requested_file
            ]

    # -----------------------------------------------------
    # 1. SHOW FIRST N ROWS
    # -----------------------------------------------------

    first_match = re.search(
        r"(?:first|top)\s+(\d+)\s+rows?",
        q
    )

    if first_match:

        n = int(
            first_match.group(1)
        )

        n = max(
            1,
            min(n, 1000)
        )

        if not csv_items:
            return None

        results = []

        for item in csv_items:

            df = item["df"]

            results.append(
                {
                    "title": (
                        f"First {n} rows — "
                        f"{item['name']}"
                    ),
                    "data": df.head(n),
                    "source": item["name"]
                }
            )

        return {
            "type": "table",
            "results": results
        }

    # -----------------------------------------------------
    # 2. SHOW LAST N ROWS
    # -----------------------------------------------------

    last_match = re.search(
        r"(?:last|bottom)\s+(\d+)\s+rows?",
        q
    )

    if last_match:

        n = int(
            last_match.group(1)
        )

        n = max(
            1,
            min(n, 1000)
        )

        if not csv_items:
            return None

        results = []

        for item in csv_items:

            df = item["df"]

            results.append(
                {
                    "title": (
                        f"Last {n} rows — "
                        f"{item['name']}"
                    ),
                    "data": df.tail(n),
                    "source": item["name"]
                }
            )

        return {
            "type": "table",
            "results": results
        }

    # -----------------------------------------------------
    # 3. SHOW SPECIFIC ROW
    # -----------------------------------------------------

    row_match = re.search(
        r"(?:show|display|give|view)?\s*"
        r"(?:the\s+)?"
        r"(?:row|record)\s*#?\s*(\d+)",
        q
    )

    if row_match:

        row_number = int(
            row_match.group(1)
        )

        if row_number <= 0:
            return None

        if not csv_items:
            return None

        results = []

        for item in csv_items:

            df = item["df"]

            if row_number > len(df):
                continue

            row_df = df.iloc[
                row_number - 1:
                row_number
            ]

            results.append(
                {
                    "title": (
                        f"Row {row_number} — "
                        f"{item['name']}"
                    ),
                    "data": row_df,
                    "source": item["name"]
                }
            )

        if results:

            return {
                "type": "table",
                "results": results
            }

    # -----------------------------------------------------
    # 4. SHOW ALL VALUES OF A COLUMN
    # -----------------------------------------------------

    value_command = any(
        phrase in q
        for phrase in [
            "show all values",
            "show every value",
            "display all values",
            "display every value",
            "give all values",
            "give me all values",
            "all values of",
            "all values in",
            "every value of",
            "every value in"
        ]
    )

    if value_command:

        if not csv_items:
            return None

        results = []

        for item in csv_items:

            df = item["df"]

            columns = find_relevant_columns(
                question,
                df
            )

            if not columns:
                continue

            column = columns[0]

            result_df = df[
                [column]
            ].copy()

            results.append(
                {
                    "title": (
                        f"All values of "
                        f"{column} — "
                        f"{item['name']}"
                    ),
                    "data": result_df,
                    "source": item["name"],
                    "field": column
                }
            )

        if results:

            return {
                "type": "column",
                "results": results
            }

    # -----------------------------------------------------
    # 5. SHOW ALL COLUMNS OF A ROW
    # -----------------------------------------------------

    if (
        "all columns of row" in q
        or "all column values of row" in q
        or "complete row" in q
        or "full row" in q
    ):

        row_match = re.search(
            r"row\s*(\d+)",
            q
        )

        if row_match:

            row_number = int(
                row_match.group(1)
            )

            results = []

            for item in csv_items:

                df = item["df"]

                if (
                    row_number > 0
                    and row_number <= len(df)
                ):

                    row_df = df.iloc[
                        row_number - 1:
                        row_number
                    ]

                    results.append(
                        {
                            "title": (
                                f"Complete row "
                                f"{row_number} — "
                                f"{item['name']}"
                            ),
                            "data": row_df,
                            "source": item["name"]
                        }
                    )

            if results:

                return {
                    "type": "table",
                    "results": results
                }

    return None


# =========================================================
# DATASET ANALYSIS
# =========================================================

def detect_operation(question):

    q = normalize_text(
        question
    )

    if any(
        x in q
        for x in [
            "average",
            "avg",
            "mean"
        ]
    ):
        return "average"

    if any(
        x in q
        for x in [
            "maximum",
            "max",
            "highest",
            "largest"
        ]
    ):
        return "max"

    if any(
        x in q
        for x in [
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
        x in q
        for x in [
            "sum",
            "total"
        ]
    ):
        return "sum"

    if any(
        x in q
        for x in [
            "unique",
            "distinct"
        ]
    ):
        return "unique"

    if any(
        x in q
        for x in [
            "missing",
            "null",
            "empty"
        ]
    ):
        return "missing"

    if (
        "data type" in q
        or "datatype" in q
        or "type of" in q
    ):
        return "dtype"

    if "first value" in q:
        return "first"

    if "last value" in q:
        return "last"

    return None


def analyze_csv_question(
    question,
    df
):

    if df is None or df.empty:
        return None, []

    q = normalize_text(
        question
    )

    # Generic rows
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
            f"The dataset has "
            f"{len(df):,} rows.",
            []
        )

    # Generic columns
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
            f"The dataset has "
            f"{len(df.columns):,} columns.",
            []
        )

    if (
        q == "columns"
        or "column names" in q
        or "what columns" in q
        or "what are the columns" in q
    ):

        names = ", ".join(
            map(str, df.columns)
        )

        return (
            f"The columns are: {names}.",
            []
        )

    columns = find_relevant_columns(
        question,
        df
    )

    # IMPORTANT:
    # No random field selection
    if not columns:
        return None, []

    column = columns[0]

    operation = detect_operation(
        question
    )

    series = df[column]

    numeric = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    info = get_column_info(
        column
    )

    if operation == "average":

        if numeric.empty:
            return (
                f"{info['name']} is not numeric.",
                [column]
            )

        return (
            f"The average "
            f"{info['name']} is "
            f"{numeric.mean():.2f}.",
            [column]
        )

    if operation == "max":

        if numeric.empty:
            return (
                f"{info['name']} is not numeric.",
                [column]
            )

        return (
            f"The maximum "
            f"{info['name']} is "
            f"{numeric.max():g}.",
            [column]
        )

    if operation == "min":

        if numeric.empty:
            return (
                f"{info['name']} is not numeric.",
                [column]
            )

        return (
            f"The minimum "
            f"{info['name']} is "
            f"{numeric.min():g}.",
            [column]
        )

    if operation == "median":

        if numeric.empty:
            return (
                f"{info['name']} is not numeric.",
                [column]
            )

        return (
            f"The median "
            f"{info['name']} is "
            f"{numeric.median():.2f}.",
            [column]
        )

    if operation == "sum":

        if numeric.empty:
            return (
                f"{info['name']} is not numeric.",
                [column]
            )

        return (
            f"The total "
            f"{info['name']} is "
            f"{numeric.sum():g}.",
            [column]
        )

    if operation == "unique":

        return (
            f"{info['name']} has "
            f"{series.nunique(dropna=True):,} "
            f"unique values.",
            [column]
        )

    if operation == "missing":

        return (
            f"{info['name']} has "
            f"{int(series.isna().sum()):,} "
            f"missing values.",
            [column]
        )

    if operation == "dtype":

        return (
            f"The data type of "
            f"{info['name']} is "
            f"{series.dtype}.",
            [column]
        )

    if operation == "first":

        values = series.dropna()

        if values.empty:
            return (
                f"No value is available for "
                f"{info['name']}.",
                [column]
            )

        return (
            f"The first "
            f"{info['name']} value is "
            f"{values.iloc[0]}.",
            [column]
        )

    if operation == "last":

        values = series.dropna()

        if values.empty:
            return (
                f"No value is available for "
                f"{info['name']}.",
                [column]
            )

        return (
            f"The last "
            f"{info['name']} value is "
            f"{values.iloc[-1]}.",
            [column]
        )

    # "what is bp"
    if any(
        x in q
        for x in [
            "what is",
            "what are",
            "tell me about",
            "explain"
        ]
    ):

        return (
            f"{info['name']} refers to "
            f"{info['meaning'].lower()}.",
            [column]
        )

    return None, []


# =========================================================
# TXT Q&A
# =========================================================

def parse_qa_dataset(text):

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
                {
                    "question": question,
                    "answer": answer
                }
            )

    return pairs


def search_qa_dataset(
    question,
    text
):

    pairs = parse_qa_dataset(
        text
    )

    if not pairs:
        return None

    q = normalize_text(
        question
    )

    # Exact
    for item in pairs:

        if q == normalize_text(
            item["question"]
        ):

            return item["answer"]

    # Fuzzy TF-IDF
    try:

        questions = [
            normalize_text(
                x["question"]
            )
            for x in pairs
        ]

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

        index = int(
            np.argmax(scores)
        )

        score = float(
            scores[index]
        )

        if score >= 0.45:

            return pairs[index][
                "answer"
            ]

    except Exception:
        pass

    return None


# =========================================================
# DOCUMENT SEARCH
# =========================================================

def split_text(
    text,
    chunk_size=900
):

    if not text:
        return []

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    words = text.split()

    chunks = []

    current = []
    length = 0

    for word in words:

        current.append(word)

        length += len(word) + 1

        if length >= chunk_size:

            chunks.append(
                " ".join(current)
            )

            current = []
            length = 0

    if current:

        chunks.append(
            " ".join(current)
        )

    return chunks


def search_document(
    question,
    text
):

    chunks = split_text(
        text
    )

    if not chunks:
        return None

    q = normalize_text(
        question
    )

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

        index = int(
            np.argmax(scores)
        )

        if scores[index] >= 0.10:

            return chunks[index]

    except Exception:
        pass

    return None


# =========================================================
# GENERAL ANSWERS
# =========================================================

GENERAL_ANSWERS = {

    "what is python":
        "Python is a high-level programming language widely used in AI, data science, web development, and automation.",

    "what is ai":
        "Artificial Intelligence (AI) is the field of computer science focused on creating systems that can perform tasks that normally require human intelligence.",

    "what is artificial intelligence":
        "Artificial Intelligence is the field of computer science focused on building systems that can learn, reason, understand information, and perform intelligent tasks.",

    "what is ml":
        "Machine Learning is a branch of AI where computers learn patterns from data and use those patterns to make predictions or decisions.",

    "what is machine learning":
        "Machine Learning allows computers to learn patterns from data without being explicitly programmed for every task.",

    "what is dl":
        "Deep Learning is a part of Machine Learning that uses multi-layer neural networks to learn complex patterns.",

    "what is deep learning":
        "Deep Learning uses neural networks with multiple layers to learn complex patterns from data.",

    "what is nlp":
        "Natural Language Processing is a field of AI that helps computers understand, process, and generate human language.",

    "what is natural language processing":
        "NLP enables computers to work with human language, including text and speech.",

    "what is computer vision":
        "Computer Vision enables computers to understand and analyze images and videos.",

    "what is csv":
        "CSV stands for Comma-Separated Values. It is a common format for storing tabular data.",

    "what is pandas":
        "Pandas is a Python library used for data manipulation and analysis.",

    "what is streamlit":
        "Streamlit is a Python framework for building interactive data and AI web applications.",

    "what is chatbot":
        "A chatbot is software that communicates with users through text or voice and provides automated responses.",

    "what is rag":
        "RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from a knowledge source and uses it to generate an answer.",

    "what is llm":
        "LLM stands for Large Language Model. It is an AI model trained on large amounts of text to understand and generate language."
}


def get_general_answer(
    question
):

    q = normalize_text(
        question
    )

    if q in GENERAL_ANSWERS:

        return GENERAL_ANSWERS[q]

    matches = difflib.get_close_matches(
        q,
        list(GENERAL_ANSWERS.keys()),
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
You are IntelliMind AI.

Answer the user's question clearly and accurately.

Rules:
- Use the provided context when relevant.
- Never invent dataset values.
- Never calculate numerical dataset answers yourself if the data context is insufficient.
- Keep the answer concise.
- Do not show internal reasoning.
- Do not mention TF-IDF, embeddings, retrieval, chunks, or internal processing.

User question:
{question}

Context:
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
    files_data
):

    parts = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        name = item.get(
            "name",
            "Unknown"
        )

        df = item.get(
            "df"
        )

        text = item.get(
            "text",
            ""
        )

        if df is not None:

            parts.append(
                f"""
FILE: {name}

Columns:
{", ".join(map(str, df.columns))}

Rows:
{len(df)}
"""
            )

        elif text:

            parts.append(
                f"""
FILE: {name}

CONTENT:
{text[:5000]}
"""
            )

    return "\n".join(
        parts
    )


# =========================================================
# MAIN QUESTION PROCESSOR
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
            "sources": [],
            "table_results": []
        }

    # -----------------------------------------------------
    # 1. DATA EXPLORER
    # -----------------------------------------------------

    explorer_result = explore_dataset(
        question,
        files_data
    )

    if explorer_result:

        return {
            "answer": "",
            "fields": [],
            "sources": [],
            "table_results":
                explorer_result["results"]
        }

    # -----------------------------------------------------
    # 2. GENERAL KNOWLEDGE
    # -----------------------------------------------------

    general = get_general_answer(
        question
    )

    if general:

        return {
            "answer": general,
            "fields": [],
            "sources": [],
            "table_results": []
        }

    # -----------------------------------------------------
    # 3. DIRECT CSV ANALYSIS
    # -----------------------------------------------------

    csv_answers = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get(
            "df"
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

    if csv_answers:

        answers = []
        fields = []
        sources = []

        for item in csv_answers:

            answers.append(
                item["answer"]
            )

            fields.extend(
                item["fields"]
            )

            sources.append(
                item["source"]
            )

        return {
            "answer": "\n\n".join(
                answers
            ),
            "fields": list(
                dict.fromkeys(
                    fields
                )
            ),
            "sources": list(
                dict.fromkeys(
                    sources
                )
            ),
            "table_results": []
        }

    # -----------------------------------------------------
    # 4. TXT Q&A
    # -----------------------------------------------------

    for item in files_data:

        if not isinstance(item, dict):
            continue

        if item.get(
            "type"
        ) != ".txt":

            continue

        text = item.get(
            "text",
            ""
        )

        if "|" not in text:
            continue

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
                ],
                "table_results": []
            }

    # -----------------------------------------------------
    # 5. PDF / DOCX
    # -----------------------------------------------------

    document_results = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        if item.get(
            "type"
        ) not in [
            ".pdf",
            ".docx"
        ]:

            continue

        text = item.get(
            "text",
            ""
        )

        result = search_document(
            question,
            text
        )

        if result:

            document_results.append(
                {
                    "source": item.get(
                        "name",
                        "Document"
                    ),
                    "text": result
                }
            )

    if document_results:

        context = "\n\n".join(
            f"Source: {x['source']}\n{x['text']}"
            for x in document_results
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
                    for x in document_results
                ],
                "table_results": []
            }

        return {
            "answer": document_results[0]["text"],
            "fields": [],
            "sources": [
                x["source"]
                for x in document_results
            ],
            "table_results": []
        }

    # -----------------------------------------------------
    # 6. GEMINI FALLBACK
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
            "sources": [],
            "table_results": []
        }

    # -----------------------------------------------------
    # 7. FINAL FALLBACK
    # -----------------------------------------------------

    return {
        "answer": (
            "I couldn't find a reliable answer. "
            "Try asking about a specific field, "
            "row, column, document topic, or dataset."
        ),
        "fields": [],
        "sources": [],
        "table_results": []
    }


# =========================================================
# SMART QUESTIONS
# =========================================================

def generate_smart_questions(
    files_data
):

    questions = []

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get(
            "df"
        )

        if df is None or df.empty:
            continue

        numeric_columns = []

        for col in df.columns:

            numeric = pd.to_numeric(
                df[col],
                errors="coerce"
            )

            if numeric.notna().sum() > 0:

                numeric_columns.append(
                    col
                )

        for col in numeric_columns[:5]:

            info = get_column_info(
                col
            )

            questions.append(
                f"What is the average {info['name']}?"
            )

        for col in numeric_columns[:2]:

            info = get_column_info(
                col
            )

            questions.append(
                f"What is the maximum {info['name']}?"
            )

            questions.append(
                f"What is the minimum {info['name']}?"
            )

        questions.append(
            f"Show first 5 rows of {item['name']}"
        )

        questions.append(
            f"Show all values of {numeric_columns[0]}"
            if numeric_columns
            else f"What columns are in {item['name']}?"
        )

    for item in files_data:

        if not isinstance(item, dict):
            continue

        if item.get(
            "type"
        ) in [
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

    return list(
        dict.fromkeys(
            questions
        )
    )[:12]


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

        if (
            names
            != st.session_state.last_uploaded_names
        ):

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

    st.markdown(
        "### 📁 Files"
    )

    if not files_data:

        st.caption(
            "No files uploaded yet."
        )

    else:

        for item in files_data:

            if not isinstance(item, dict):
                continue

            name = item.get(
                "name",
                "Unknown"
            )

            df = item.get(
                "df"
            )

            with st.expander(
                f"📄 {name}"
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
                            f"`{col}` → {info['name']}"
                        )

                        with st.expander(
                            "Meaning"
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

    st.markdown(
        "### 📚 Fields"
    )

    shown = set()

    for item in files_data:

        if not isinstance(item, dict):
            continue

        df = item.get(
            "df"
        )

        if df is None:
            continue

        for col in df.columns:

            key = column_key(
                col
            )

            if key in shown:
                continue

            shown.add(key)

            info = get_column_info(
                col
            )

            st.markdown(
                f"`{col}` → {info['name']}"
            )

            if len(shown) >= 8:
                break


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
        "👈 Upload CSV, TXT, PDF or DOCX files to get started."
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown(
            "### 📊 Data Analysis"
        )

        st.caption(
            "Analyze columns, values, rows and statistics."
        )

    with c2:

        st.markdown(
            "### 📄 Documents"
        )

        st.caption(
            "Ask questions from TXT, PDF and DOCX files."
        )

    with c3:

        st.markdown(
            "### 🧠 AI Assistant"
        )

        st.caption(
            "Ask questions naturally."
        )


# =========================================================
# OVERVIEW
# =========================================================

else:

    total_files = len(
        files_data
    )

    datasets = [
        x
        for x in files_data
        if isinstance(x, dict)
        and x.get("df") is not None
    ]

    documents = total_files - len(
        datasets
    )

    total_rows = sum(
        len(x["df"])
        for x in datasets
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
            len(datasets)
        )

    with c3:
        st.metric(
            "Documents",
            documents
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

    if st.session_state.smart_questions:

        cols = st.columns(3)

        for i, question in enumerate(
            st.session_state.smart_questions
        ):

            with cols[i % 3]:

                if st.button(
                    question,
                    key=f"smart_{i}",
                    use_container_width=True
                ):

                    st.session_state.selected_question = (
                        question
                    )

                    st.rerun()


# =========================================================
# CHAT HISTORY
# =========================================================

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

            st.write(
                content
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            if content:

                st.write(
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

            tables = message.get(
                "table_results",
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

            if tables:

                for table in tables:

                    st.markdown(
                        f"**{table['title']}**"
                    )

                    data = table["data"]

                    if isinstance(
                        data,
                        pd.Series
                    ):

                        data = data.to_frame().T

                    st.dataframe(
                        data,
                        use_container_width=True,
                        hide_index=False
                    )


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask about your files..."
)

if st.session_state.selected_question:

    question = (
        st.session_state.selected_question
    )

    st.session_state.selected_question = None


# =========================================================
# RUN QUESTION
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
            "content": result.get(
                "answer",
                ""
            ),
            "fields": result.get(
                "fields",
                []
            ),
            "sources": result.get(
                "sources",
                []
            ),
            "table_results": result.get(
                "table_results",
                []
            )
        }
    )

    st.rerun()
