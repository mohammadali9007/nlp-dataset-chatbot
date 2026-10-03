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

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "selected_question" not in st.session_state:
    st.session_state.selected_question = None


# =========================================================
# COLUMN FULL FORM DICTIONARY
# =========================================================

COLUMN_INFO = {

    # Diabetes
    "pregnancies": ("Pregnancies", "Number of pregnancies"),
    "glucose": ("Glucose", "Plasma glucose concentration"),
    "bloodpressure": ("Blood Pressure", "Diastolic blood pressure"),
    "skinthickness": ("Skin Thickness", "Triceps skin fold thickness"),
    "insulin": ("Insulin", "2-hour serum insulin"),
    "bmi": ("BMI", "Body Mass Index"),
    "diabetespedigreefunction": (
        "Diabetes Pedigree Function",
        "Diabetes hereditary risk score"
    ),
    "age": ("Age", "Age of the person"),
    "outcome": ("Outcome", "Diabetes outcome/classification"),

    # Kidney
    "id": ("ID", "Unique record identifier"),
    "bp": ("Blood Pressure", "Blood pressure"),
    "sg": ("Specific Gravity", "Urine specific gravity"),
    "al": ("Albumin", "Albumin level"),
    "su": ("Sugar", "Urine sugar level"),
    "rbc": ("Red Blood Cells", "Red blood cell condition"),
    "pc": ("Pus Cell", "Pus cell condition"),
    "pcc": ("Pus Cell Clumps", "Presence of pus cell clumps"),
    "ba": ("Bacteria", "Presence of bacteria"),
    "bgr": ("Blood Glucose Random", "Random blood glucose level"),
    "bu": ("Blood Urea", "Blood urea level"),
    "sc": ("Serum Creatinine", "Serum creatinine level"),
    "sod": ("Sodium", "Serum sodium level"),
    "pot": ("Potassium", "Serum potassium level"),
    "hemo": ("Hemoglobin", "Hemoglobin level"),
    "pcv": ("Packed Cell Volume", "Packed cell volume"),
    "wc": ("White Blood Cell Count", "White blood cell count"),
    "rc": ("Red Blood Cell Count", "Red blood cell count"),
    "htn": ("Hypertension", "Whether hypertension is present"),
    "dm": ("Diabetes Mellitus", "Whether diabetes mellitus is present"),
    "cad": ("Coronary Artery Disease", "Whether coronary artery disease is present"),
    "appet": ("Appetite", "Patient appetite"),
    "pe": ("Pedal Edema", "Presence of pedal edema"),
    "ane": ("Anemia", "Presence of anemia"),
    "classification": (
        "Classification",
        "Final disease classification"
    ),

    # Common datasets
    "name": ("Name", "Name of the person"),
    "gender": ("Gender", "Gender of the person"),
    "sex": ("Sex", "Sex of the person"),
    "salary": ("Salary", "Salary amount"),
    "income": ("Income", "Income amount"),
    "department": ("Department", "Department name"),
    "education": ("Education", "Education level"),
    "experience": ("Experience", "Years of experience"),
    "height": ("Height", "Height measurement"),
    "weight": ("Weight", "Weight measurement"),
    "city": ("City", "City name"),
    "country": ("Country", "Country name"),
    "email": ("Email", "Email address"),
    "phone": ("Phone", "Phone number"),
    "date": ("Date", "Date value"),
    "time": ("Time", "Time value"),
    "price": ("Price", "Price amount"),
    "quantity": ("Quantity", "Quantity value"),
    "category": ("Category", "Category name"),
    "rating": ("Rating", "Rating value"),
    "score": ("Score", "Score value"),
    "marks": ("Marks", "Marks obtained"),
    "gpa": ("GPA", "Grade Point Average"),
}


# =========================================================
# TEXT HELPERS
# =========================================================

def normalize_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_column_name(column):
    return str(column).strip()


def column_key(column):
    return re.sub(r"[^a-z0-9]", "", str(column).lower())


def get_column_info(column):

    key = column_key(column)

    if key in COLUMN_INFO:
        return COLUMN_INFO[key]

    # Automatic fallback
    pretty = re.sub(r"[_\-]+", " ", str(column))
    pretty = re.sub(r"([a-z])([A-Z])", r"\1 \2", pretty)
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
        return 0

    if a == b:
        return 1.0

    ratio = difflib.SequenceMatcher(
        None,
        a,
        b
    ).ratio()

    return ratio


# =========================================================
# CSV READER
# =========================================================

def read_csv_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        df = pd.read_csv(uploaded_file)

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
            "text": f"Could not read CSV: {e}",
            "columns": []
        }


# =========================================================
# TXT READER
# =========================================================

def read_txt_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        content = uploaded_file.read()

        if isinstance(content, bytes):
            content = content.decode(
                "utf-8",
                errors="ignore"
            )

        return {
            "name": uploaded_file.name,
            "type": "txt",
            "data": None,
            "text": content,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"Could not read TXT: {e}",
            "columns": []
        }


# =========================================================
# PDF READER
# =========================================================

def read_pdf_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        reader = PdfReader(uploaded_file)

        pages = []

        for page in reader.pages:

            text = page.extract_text()

            if text:
                pages.append(text)

        content = "\n".join(pages)

        return {
            "name": uploaded_file.name,
            "type": "pdf",
            "data": None,
            "text": content,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"Could not read PDF: {e}",
            "columns": []
        }


# =========================================================
# DOCX READER
# =========================================================

def read_docx_file(uploaded_file):

    try:

        uploaded_file.seek(0)

        document = Document(uploaded_file)

        paragraphs = []

        for paragraph in document.paragraphs:

            if paragraph.text.strip():
                paragraphs.append(paragraph.text)

        content = "\n".join(paragraphs)

        return {
            "name": uploaded_file.name,
            "type": "docx",
            "data": None,
            "text": content,
            "columns": []
        }

    except Exception as e:

        return {
            "name": uploaded_file.name,
            "type": "error",
            "data": None,
            "text": f"Could not read DOCX: {e}",
            "columns": []
        }


# =========================================================
# FILE PROCESSOR
# =========================================================

def process_uploaded_files(uploaded_files):

    files = []

    for uploaded_file in uploaded_files:

        name = uploaded_file.name.lower()

        if name.endswith(".csv"):
            item = read_csv_file(uploaded_file)

        elif name.endswith(".txt"):
            item = read_txt_file(uploaded_file)

        elif name.endswith(".pdf"):
            item = read_pdf_file(uploaded_file)

        elif name.endswith(".docx"):
            item = read_docx_file(uploaded_file)

        else:
            continue

        files.append(item)

    return files


# =========================================================
# TXT Q&A SEARCH
# =========================================================

def parse_qa_text(text):

    qa_pairs = []

    lines = text.splitlines()

    for line in lines:

        if "|" not in line:
            continue

        parts = line.split("|", 1)

        question = parts[0].strip()
        answer = parts[1].strip()

        if question and answer:

            qa_pairs.append(
                {
                    "question": question,
                    "answer": answer
                }
            )

    return qa_pairs


def search_qa_dataset(question, text):

    qa_pairs = parse_qa_text(text)

    if not qa_pairs:
        return None

    best_answer = None
    best_score = 0

    for item in qa_pairs:

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
# DOCUMENT SEARCH
# =========================================================

def split_text(text, chunk_size=700):

    text = str(text)

    words = text.split()

    chunks = []

    current = []

    count = 0

    for word in words:

        current.append(word)
        count += len(word)

        if count >= chunk_size:

            chunks.append(" ".join(current))

            current = []
            count = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


def search_documents(question, files):

    documents = []

    for file in files:

        if file["type"] in ["txt", "pdf", "docx"]:

            text = file.get("text", "")

            if not text:
                continue

            chunks = split_text(text)

            for chunk in chunks:

                documents.append(
                    {
                        "file": file["name"],
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

    ranked = np.argsort(scores)[::-1]

    results = []

    for index in ranked[:5]:

        if scores[index] < 0.05:
            continue

        results.append(
            {
                "file": documents[index]["file"],
                "text": documents[index]["text"],
                "score": float(scores[index])
            }
        )

    return results


# =========================================================
# CSV QUESTION ANALYSIS
# =========================================================

def find_relevant_columns(question, df):

    question_normalized = normalize_text(question)

    matches = []

    for column in df.columns:

        key = column_key(column)

        name = normalize_text(column)

        score = 0

        if name in question_normalized:
            score = 1

        elif key in question_normalized.replace(" ", ""):
            score = 0.95

        else:

            score = difflib.SequenceMatcher(
                None,
                question_normalized,
                name
            ).ratio()

        # Word matching
        column_words = set(name.split())
        question_words = set(
            question_normalized.split()
        )

        overlap = len(
            column_words.intersection(question_words)
        )

        if overlap > 0:
            score = max(score, 0.75)

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
# CSV DIRECT ANSWER
# =========================================================

def analyze_csv_question(question, df):

    q = normalize_text(question)

    columns = find_relevant_columns(
        question,
        df
    )

    if not columns:
        return None, []


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
            f"This dataset contains **{len(df):,} rows**.",
            []
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
            f"This dataset contains **{len(df.columns)} columns**.",
            []
        )


    # -----------------------------------------------------
    # COLUMN LIST
    # -----------------------------------------------------

    if (
        "column names" in q
        or "columns name" in q
        or "what are the columns" in q
        or "list columns" in q
    ):

        names = ", ".join(
            str(c)
            for c in df.columns
        )

        return (
            f"The columns are: **{names}**",
            []
        )


    # -----------------------------------------------------
    # AVERAGE
    # -----------------------------------------------------

    if (
        "average" in q
        or "mean" in q
    ):

        column = columns[0]

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

        column = columns[0]

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

        column = columns[0]

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

        column = columns[0]

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
    # UNIQUE VALUES
    # -----------------------------------------------------

    if (
        "unique" in q
        or "distinct" in q
    ):

        column = columns[0]

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
    ):

        column = columns[0]

        count = df[column].isna().sum()

        return (
            f"**{column}** has "
            f"**{count:,} missing values**.",
            [column]
        )


    # -----------------------------------------------------
    # COLUMN MEANING
    # -----------------------------------------------------

    if (
        "what is" in q
        or "meaning" in q
        or "means" in q
        or "define" in q
        or "what does" in q
    ):

        column = columns[0]

        full_name, meaning = get_column_info(
            column
        )

        return (
            f"**{column}** means "
            f"**{full_name}**. "
            f"{meaning}.",
            [column]
        )


    return None, []


# =========================================================
# BUILT-IN GENERAL KNOWLEDGE
# =========================================================

GENERAL_ANSWERS = {

    "what is ai":
        "AI (Artificial Intelligence) is a technology that allows computers to perform tasks that normally require human intelligence.",

    "what is machine learning":
        "Machine Learning is a branch of AI where computers learn patterns from data and use those patterns to make predictions or decisions.",

    "what is ml":
        "Machine Learning (ML) is a branch of AI that allows computers to learn patterns from data without being explicitly programmed for every task.",

    "what is deep learning":
        "Deep Learning is a part of Machine Learning that uses neural networks with multiple layers to learn complex patterns from data.",

    "what is nlp":
        "NLP (Natural Language Processing) is a field of AI that helps computers understand, process, and generate human language.",

    "what is computer vision":
        "Computer Vision is a field of AI that allows computers to understand and analyze images and videos.",

    "what is python":
        "Python is a high-level programming language widely used in web development, automation, data science, machine learning, and AI.",

    "what is csv":
        "CSV stands for Comma-Separated Values. It is a simple file format used to store tabular data.",

}


def get_general_answer(question):

    q = normalize_text(question)

    for key, answer in GENERAL_ANSWERS.items():

        score = text_similarity(
            q,
            key
        )

        if score >= 0.82:
            return answer

    return None


# =========================================================
# GEMINI
# =========================================================

def generate_ai_answer(question, context):

    if genai is None:
        return None

    api_key = None

    try:

        if "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets[
                "GEMINI_API_KEY"
            ]

    except Exception:
        pass

    if not api_key:
        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

    if not api_key:
        return None

    try:

        client = genai.Client(
            api_key=api_key
        )

        prompt = f"""
You are IntelliMind AI, a professional multi-file data assistant.

Answer the user's question using the provided file context.

Rules:
1. Use the uploaded files when relevant.
2. Do not invent numbers.
3. If a CSV contains the answer, use the actual data.
4. If the context does not contain the answer, say that clearly.
5. Give a direct and easy-to-understand answer.
6. Do not show internal reasoning.
7. Do not show raw matching rows unless the user explicitly asks.
8. Keep the answer concise but useful.

USER QUESTION:
{question}

FILE CONTEXT:
{context}
"""

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
# BUILD ALL FILE CONTEXT
# =========================================================

def build_file_context(question, files):

    context_parts = []

    # CSV context
    for file in files:

        if file["type"] != "csv":
            continue

        df = file["data"]

        answer, relevant_columns = analyze_csv_question(
            question,
            df
        )

        if answer:

            context_parts.append(
                f"""
FILE: {file['name']}

CSV DIRECT RESULT:
{answer}

RELEVANT COLUMNS:
{", ".join(map(str, relevant_columns))}
"""
            )

    # Document context
    document_results = search_documents(
        question,
        files
    )

    for result in document_results:

        context_parts.append(
            f"""
FILE: {result['file']}

DOCUMENT CONTENT:
{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# FIND ANSWER
# =========================================================

def process_question(question):

    question = question.strip()

    if not question:
        return (
            "Please enter a question.",
            [],
            []
        )

    # -----------------------------------------------------
    # GENERAL BUILT-IN ANSWER
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


    # -----------------------------------------------------
    # TXT EXACT / FUZZY QA
    # -----------------------------------------------------

    for file in st.session_state.files_data:

        if file["type"] != "txt":
            continue

        answer = search_qa_dataset(
            question,
            file["text"]
        )

        if answer:

            return (
                answer,
                [],
                [file["name"]]
            )


    # -----------------------------------------------------
    # DIRECT CSV ANSWER
    # -----------------------------------------------------

    csv_answers = []

    relevant_fields = []

    source_files = []

    for file in st.session_state.files_data:

        if file["type"] != "csv":
            continue

        answer, fields = analyze_csv_question(
            question,
            file["data"]
        )

        if answer:

            csv_answers.append(
                f"{file['name']}: {answer}"
            )

            relevant_fields.extend(
                fields
            )

            source_files.append(
                file["name"]
            )


    if len(csv_answers) == 1:

        return (
            csv_answers[0].split(": ", 1)[1],
            list(dict.fromkeys(relevant_fields)),
            source_files
        )


    if len(csv_answers) > 1:

        combined = "\n\n".join(
            csv_answers
        )

        return (
            combined,
            list(dict.fromkeys(relevant_fields)),
            source_files
        )


    # -----------------------------------------------------
    # ALL FILE CONTEXT
    # -----------------------------------------------------

    context = build_file_context(
        question,
        st.session_state.files_data
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

            for file in st.session_state.files_data:

                if file["name"] in context:
                    sources.append(
                        file["name"]
                    )

            return (
                ai_answer,
                [],
                sources
            )


    # -----------------------------------------------------
    # FALLBACK
    # -----------------------------------------------------

    return (
        "I could not find a reliable answer in the uploaded files. "
        "Try asking about a specific column, value, topic, or document content.",
        [],
        []
    )


# =========================================================
# DYNAMIC QUESTION SUGGESTIONS
# =========================================================

def generate_smart_questions(files):

    questions = []

    for file in files:

        if file["type"] != "csv":
            continue

        df = file["data"]

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        for column in numeric_columns[:3]:

            questions.append(
                f"What is the average {column}?"
            )

        if len(df.columns) > 0:

            questions.append(
                f"How many rows are in {file['name']}?"
            )

        break

    # Document suggestions

    for file in files:

        if file["type"] in [
            "txt",
            "pdf",
            "docx"
        ]:

            questions.append(
                f"What is this document about?"
            )

            break

    # Remove duplicates

    unique = []

    for q in questions:

        if q not in unique:
            unique.append(q)

    return unique[:6]


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        "# 🧠 IntelliMind AI"
    )

    st.caption(
        "Your Multi-File Data Copilot"
    )

    st.divider()

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

    if uploaded_files:

        st.session_state.files_data = (
            process_uploaded_files(
                uploaded_files
            )
        )

    st.divider()


    # =====================================================
    # FILE EXPLORER
    # =====================================================

    st.markdown(
        "### 📁 File Explorer"
    )

    if st.session_state.files_data:

        for index, file in enumerate(
            st.session_state.files_data
        ):

            file_type = file["type"]

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
            # CLICKABLE FILE NAME
            # ------------------------------------------------

            with st.expander(
                f"{icon} {file['name']}",
                expanded=False
            ):

                # CSV
                if file_type == "csv":

                    df = file["data"]

                    st.caption(
                        f"{len(df):,} rows • "
                        f"{len(df.columns)} columns"
                    )

                    st.markdown(
                        "#### 🧩 All Columns"
                    )

                    for col_index, column in enumerate(
                        df.columns,
                        start=1
                    ):

                        full_name, meaning = (
                            get_column_info(column)
                        )

                        st.markdown(
                            f"**{col_index}. `{column}`**"
                        )

                        st.caption(
                            f"Full name: {full_name}"
                        )

                        # Detailed meaning only when opened
                        with st.expander(
                            "Meaning",
                            expanded=False
                        ):

                            st.write(
                                meaning
                            )


                # TXT / PDF / DOCX
                elif file_type in [
                    "txt",
                    "pdf",
                    "docx"
                ]:

                    text_content = file.get(
                        "text",
                        ""
                    )

                    if file_type == "txt":

                        qa_count = len(
                            parse_qa_text(
                                text_content
                            )
                        )

                        st.caption(
                            f"{qa_count:,} Q&A entries"
                        )

                    else:

                        word_count = len(
                            text_content.split()
                        )

                        st.caption(
                            f"{word_count:,} words"
                        )

                    st.markdown(
                        "#### 📄 File Content"
                    )

                    preview = text_content[:2500]

                    if preview:

                        st.text(
                            preview
                        )

                        if len(text_content) > 2500:

                            st.caption(
                                "Preview only. Ask questions to search the full file."
                            )

                    else:

                        st.warning(
                            "No readable text found."
                        )


    else:

        st.info(
            "Upload one or more files to begin."
        )


    st.divider()


    # =====================================================
    # FIELD DICTIONARY
    # =====================================================

    if st.session_state.files_data:

        st.markdown(
            "### 📚 Fields"
        )

        shown_columns = []

        for file in st.session_state.files_data:

            if file["type"] != "csv":
                continue

            for column in file["columns"]:

                if column not in shown_columns:
                    shown_columns.append(column)

        for column in shown_columns[:8]:

            full_name, _ = get_column_info(
                column
            )

            st.markdown(
                f"`{column}` → {full_name}"
            )

        if len(shown_columns) > 8:

            with st.expander(
                "View all fields"
            ):

                for column in shown_columns[8:]:

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
    "Analyze multiple datasets and documents in one intelligent workspace."
)


# =========================================================
# OVERVIEW
# =========================================================

files_count = len(
    st.session_state.files_data
)

csv_count = sum(
    1
    for f in st.session_state.files_data
    if f["type"] == "csv"
)

document_count = sum(
    1
    for f in st.session_state.files_data
    if f["type"] in [
        "txt",
        "pdf",
        "docx"
    ]
)

total_rows = sum(
    len(f["data"])
    for f in st.session_state.files_data
    if f["type"] == "csv"
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "📂 Files",
        files_count
    )

with col2:
    st.metric(
        "📊 Datasets",
        csv_count
    )

with col3:
    st.metric(
        "📄 Documents",
        document_count
    )

with col4:
    st.metric(
        "🔢 Total Rows",
        f"{total_rows:,}"
    )


# =========================================================
# SMART QUESTIONS
# =========================================================

if st.session_state.files_data:

    st.subheader(
        "✨ Smart Questions"
    )

    smart_questions = generate_smart_questions(
        st.session_state.files_data
    )

    if smart_questions:

        question_columns = st.columns(3)

        for i, question in enumerate(
            smart_questions
        ):

            with question_columns[
                i % 3
            ]:

                if st.button(
                    question,
                    key=f"smart_{i}",
                    use_container_width=True
                ):

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


# =========================================================
# CHAT
# =========================================================

st.subheader(
    "💬 Conversation"
)


# Show previous messages

for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message(
            "user"
        ):

            st.write(
                message["content"]
            )

    else:

        with st.chat_message(
            "assistant"
        ):

            st.markdown(
                message["content"]
            )

            fields = message.get(
                "fields",
                []
            )

            sources = message.get(
                "sources",
                []
            )

            # Short relevant field information
            if fields:

                field_text = []

                for field in fields:

                    full_name, _ = (
                        get_column_info(
                            field
                        )
                    )

                    field_text.append(
                        f"`{field}` → {full_name}"
                    )

                st.caption(
                    " • ".join(field_text)
                )

            # Compact source
            if sources:

                st.caption(
                    "Source: " +
                    ", ".join(sources)
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
