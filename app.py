import streamlit as st
import pandas as pd
import numpy as np
import os
import re
import difflib
from io import StringIO

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    color: #777;
    margin-bottom: 25px;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.18);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from your CSV and TXT knowledge base.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GEMINI
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY", "")

client = None

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        client = None


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
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload CSV / TXT files",
        type=["csv", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:

        for file in uploaded_files:

            filename = file.name

            try:

                # -----------------------------------------
                # CSV
                # -----------------------------------------

                if filename.lower().endswith(".csv"):

                    df = pd.read_csv(file)

                    st.session_state.csv_data[
                        filename
                    ] = df


                # -----------------------------------------
                # TXT
                # -----------------------------------------

                elif filename.lower().endswith(".txt"):

                    text = file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    st.session_state.txt_data[
                        filename
                    ] = text

            except Exception as e:

                st.error(
                    f"Could not read {filename}: {e}"
                )


    st.divider()

    # CSV files
    if st.session_state.csv_data:

        st.subheader("📊 CSV Files")

        for name, dataframe in st.session_state.csv_data.items():

            st.write(
                f"• {name} "
                f"({len(dataframe)} rows)"
            )


    # TXT files
    if st.session_state.txt_data:

        st.subheader("📄 TXT Files")

        for name in st.session_state.txt_data:

            st.write(
                f"• {name}"
            )


    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# NO FILE
# =========================================================

if (
    not st.session_state.csv_data
    and
    not st.session_state.txt_data
):

    st.info(
        "👈 Upload CSV or TXT files from the sidebar."
    )

    st.markdown("### Example questions")

    c1, c2, c3 = st.columns(3)

    with c1:

        st.markdown("""
        **📊 CSV**

        - What is the average glucose?
        - How many rows?
        - Highest BMI?
        - Show diabetic patients.
        """)

    with c2:

        st.markdown("""
        **📄 TXT**

        - What is NLP?
        - Explain machine learning.
        - What does the document say?
        - Summarize the file.
        """)

    with c3:

        st.markdown("""
        **🔀 Both**

        - Compare the information.
        - Explain this data.
        - Find relevant information.
        """)

    st.stop()


# =========================================================
# FILE COUNTS
# =========================================================

total_csv = len(
    st.session_state.csv_data
)

total_txt = len(
    st.session_state.txt_data
)

total_files = total_csv + total_txt


c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "📁 Files",
        total_files
    )

with c2:
    st.metric(
        "📊 CSV",
        total_csv
    )

with c3:
    st.metric(
        "📄 TXT",
        total_txt
    )


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
        r"[^a-zA-Z0-9\s]",
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
# FUZZY WORD MATCH
# =========================================================

def fuzzy_match(word, choices, cutoff=0.70):

    if not choices:
        return None

    word = normalize_text(word)

    choices_normalized = {
        normalize_text(str(x)): x
        for x in choices
    }

    match = difflib.get_close_matches(
        word,
        list(choices_normalized.keys()),
        n=1,
        cutoff=cutoff
    )

    if match:

        return choices_normalized[
            match[0]
        ]

    return None


# =========================================================
# GET ALL CSV COLUMNS
# =========================================================

def get_all_columns():

    columns = []

    for df in st.session_state.csv_data.values():

        for col in df.columns:

            if col not in columns:
                columns.append(col)

    return columns


# =========================================================
# FIND RELEVANT CSV COLUMNS
# =========================================================

def find_relevant_columns(question):

    question_normalized = normalize_text(
        question
    )

    all_columns = get_all_columns()

    relevant = []

    # -----------------------------------------
    # Exact / partial matching
    # -----------------------------------------

    for col in all_columns:

        col_normalized = normalize_text(
            col
        )

        if col_normalized in question_normalized:

            relevant.append(col)

            continue

        words = col_normalized.split()

        for word in words:

            if len(word) >= 3:

                if word in question_normalized:

                    relevant.append(col)

                    break


    # -----------------------------------------
    # Fuzzy matching
    # -----------------------------------------

    question_words = question_normalized.split()

    for word in question_words:

        if len(word) < 3:
            continue

        matched = fuzzy_match(
            word,
            all_columns,
            cutoff=0.65
        )

        if matched and matched not in relevant:

            relevant.append(matched)


    # -----------------------------------------
    # Common diabetes synonyms
    # -----------------------------------------

    synonym_map = {

        "sugar": [
            "glucose",
            "blood glucose",
            "blood sugar"
        ],

        "bloodsugar": [
            "glucose"
        ],

        "pressure": [
            "bloodpressure",
            "blood pressure"
        ],

        "weight": [
            "weight"
        ],

        "pregnancy": [
            "pregnancies"
        ],

        "diabetic": [
            "outcome",
            "diabetes"
        ],

        "diabetes": [
            "outcome",
            "diabetic"
        ],

        "bmi": [
            "bmi"
        ],

        "age": [
            "age"
        ],

        "insulin": [
            "insulin"
        ]
    }

    for key, possible_columns in synonym_map.items():

        if key in question_normalized:

            for col in all_columns:

                col_normalized = normalize_text(
                    col
                )

                for possible in possible_columns:

                    if normalize_text(
                        possible
                    ) in col_normalized:

                        if col not in relevant:
                            relevant.append(col)


    return relevant


# =========================================================
# GET NUMERIC COLUMNS
# =========================================================

def get_numeric_columns():

    numeric = []

    for df in st.session_state.csv_data.values():

        for col in df.select_dtypes(
            include=np.number
        ).columns:

            if col not in numeric:

                numeric.append(col)

    return numeric


# =========================================================
# FIND RELEVANT TXT CHUNKS
# =========================================================

def split_text(text, chunk_size=1200):

    words = text.split()

    chunks = []

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


def search_txt(question, top_k=5):

    documents = []

    sources = []

    for filename, text in st.session_state.txt_data.items():

        chunks = split_text(text)

        for chunk in chunks:

            documents.append(chunk)

            sources.append(filename)


    if not documents:

        return []


    # -----------------------------------------
    # TF-IDF
    # -----------------------------------------

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
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

        ranked = np.argsort(
            scores
        )[::-1]

        results = []

        for index in ranked[:top_k]:

            if scores[index] > 0.03:

                results.append({
                    "text": documents[index],
                    "source": sources[index],
                    "score": float(scores[index])
                })

        return results

    except Exception:

        return []


# =========================================================
# CSV RELEVANT DATA
# =========================================================

def search_csv(question):

    relevant_columns = find_relevant_columns(
        question
    )

    results = []

    for filename, df in st.session_state.csv_data.items():

        columns = relevant_columns.copy()

        # Keep only columns existing in this dataframe
        columns = [
            col for col in columns
            if col in df.columns
        ]

        # If no matching columns,
        # use numeric columns
        if not columns:

            numeric = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            columns = numeric[:8]

        if columns:

            data = df[columns].copy()

            results.append({
                "file": filename,
                "columns": columns,
                "data": data
            })

    return results


# =========================================================
# DIRECT NUMERIC ANALYSIS
# =========================================================

def numeric_analysis(question):

    q = normalize_text(
        question
    )

    relevant_columns = find_relevant_columns(
        question
    )

    answers = []

    # =========================================
    # ROW COUNT
    # =========================================

    if (
        "how many rows" in q
        or "total rows" in q
        or "number of rows" in q
        or "how many patient" in q
        or "how many patients" in q
    ):

        total = sum(
            len(df)
            for df in st.session_state.csv_data.values()
        )

        answers.append(
            f"The dataset contains **{total:,} rows/records**."
        )


    # =========================================
    # COLUMN COUNT
    # =========================================

    if (
        "how many columns" in q
        or "number of columns" in q
    ):

        total_columns = len(
            get_all_columns()
        )

        answers.append(
            f"The dataset contains **{total_columns} columns**."
        )


    # =========================================
    # AVERAGE
    # =========================================

    if (
        "average" in q
        or "mean" in q
        or "avg" in q
    ):

        numeric_columns = [
            col
            for col in relevant_columns
            if col in get_numeric_columns()
        ]

        # If no exact column found,
        # fuzzy match numeric columns
        if not numeric_columns:

            numeric_columns = []

            for word in q.split():

                match = fuzzy_match(
                    word,
                    get_numeric_columns(),
                    cutoff=0.65
                )

                if match:

                    numeric_columns.append(
                        match
                    )


        for column in numeric_columns:

            for filename, df in st.session_state.csv_data.items():

                if column in df.columns:

                    value = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).mean()

                    if not pd.isna(value):

                        answers.append(
                            f"The average **{column}** is "
                            f"**{value:.2f}**."
                        )


    # =========================================
    # MAX / HIGHEST
    # =========================================

    if (
        "highest" in q
        or "maximum" in q
        or "max" in q
        or "largest" in q
    ):

        numeric_columns = [
            col
            for col in relevant_columns
            if col in get_numeric_columns()
        ]

        for column in numeric_columns:

            for filename, df in st.session_state.csv_data.items():

                if column in df.columns:

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

                    value = values.max()

                    if not pd.isna(value):

                        answers.append(
                            f"The highest **{column}** is "
                            f"**{value:.2f}**."
                        )


    # =========================================
    # MIN / LOWEST
    # =========================================

    if (
        "lowest" in q
        or "minimum" in q
        or "min" in q
        or "smallest" in q
    ):

        numeric_columns = [
            col
            for col in relevant_columns
            if col in get_numeric_columns()
        ]

        for column in numeric_columns:

            for filename, df in st.session_state.csv_data.items():

                if column in df.columns:

                    values = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    )

                    value = values.min()

                    if not pd.isna(value):

                        answers.append(
                            f"The lowest **{column}** is "
                            f"**{value:.2f}**."
                        )


    # =========================================
    # MISSING VALUES
    # =========================================

    if (
        "missing" in q
        or "null" in q
        or "empty" in q
    ):

        for filename, df in st.session_state.csv_data.items():

            missing = df.isna().sum()

            missing = missing[
                missing > 0
            ]

            if len(missing) > 0:

                for col, value in missing.items():

                    answers.append(
                        f"**{col}** has "
                        f"**{int(value)} missing values**."
                    )

            else:

                answers.append(
                    f"**{filename}** has no missing values."
                )


    if answers:

        return "\n\n".join(
            dict.fromkeys(answers)
        )

    return None


# =========================================================
# BUILD CSV CONTEXT
# =========================================================

def build_csv_context(question):

    results = search_csv(
        question
    )

    context_parts = []

    for result in results:

        filename = result["file"]

        data = result["data"]

        # Limit rows
        data = data.head(50)

        csv_text = data.to_csv(
            index=False
        )

        context_parts.append(
            f"""
SOURCE FILE: {filename}

COLUMNS:
{", ".join(result["columns"])}

DATA:
{csv_text}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# BUILD TXT CONTEXT
# =========================================================

def build_txt_context(question):

    results = search_txt(
        question,
        top_k=5
    )

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
SOURCE: {result["source"]}
RELEVANCE SCORE: {result["score"]:.3f}

TEXT:
{result["text"]}
"""
        )

    return "\n".join(
        context_parts
    )


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_answer(
    question,
    csv_context,
    txt_context
):

    if client is None:

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` "
            "to Streamlit Secrets."
        )


    prompt = f"""
You are IntelliMind AI, a reliable knowledge-base
and dataset question-answering assistant.

The user has uploaded CSV and/or TXT files.

Your task is to answer the user's question using
the provided information.

===============================
IMPORTANT RULES
===============================

1. Use the provided CSV/TXT information first.

2. NEVER invent a value that is not supported by
   the provided data.

3. If the question has spelling mistakes, understand
   the likely intended meaning from the context.

4. If the user writes a slightly incorrect word such as:

   "glocose" -> glucose
   "avarage" -> average
   "diabtes" -> diabetes
   "patint" -> patient

   try to understand the intended question.

5. For CSV numerical questions, trust the calculated
   results when they are provided.

6. If the dataset does not contain enough information,
   clearly say that the information is not available.

7. Do not pretend that information exists if it does not.

8. Keep answers simple and direct.

9. If useful, show a short explanation.

10. If the dataset is medical/diabetes related,
    describe the dataset and statistics only.
    Do not diagnose a person or prescribe treatment.

11. If the question is about a specific patient,
    only use information present in the dataset.

12. If CSV and TXT both contain relevant information,
    combine them carefully.

===============================
USER QUESTION
===============================

{question}

===============================
CSV CONTEXT
===============================

{csv_context}

===============================
TXT CONTEXT
===============================

{txt_context}

===============================
FINAL ANSWER
===============================

Answer the user's question naturally.
"""


    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "⚠️ AI generation error:\n\n"
            f"`{str(e)}`"
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


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask anything about your CSV or TXT..."
)


if question:

    # -----------------------------------------
    # Save user question
    # -----------------------------------------

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # -----------------------------------------
    # Search
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your knowledge base..."
        ):

            # Direct numerical calculation
            direct_answer = numeric_analysis(
                question
            )

            # TXT search
            txt_context = build_txt_context(
                question
            )

            # CSV search
            csv_context = build_csv_context(
                question
            )


        # -----------------------------------------
        # Decide answer
        # -----------------------------------------

        if (
            direct_answer
            and not txt_context
        ):

            answer = direct_answer

        else:

            with st.spinner(
                "🤖 Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    csv_context,
                    txt_context
                )


                # If Gemini unavailable,
                # use direct answer
                if (
                    answer.startswith("⚠️")
                    and direct_answer
                ):

                    answer = direct_answer


        # -----------------------------------------
        # Display
        # -----------------------------------------

        st.markdown(answer)


        # -----------------------------------------
        # Relevant CSV data
        # -----------------------------------------

        csv_results = search_csv(
            question
        )

        if csv_results:

            with st.expander(
                "📊 View relevant CSV data"
            ):

                for result in csv_results:

                    st.caption(
                        f"Source: {result['file']}"
                    )

                    st.dataframe(
                        result["data"].head(20),
                        use_container_width=True
                    )


        # -----------------------------------------
        # Relevant TXT source
        # -----------------------------------------

        txt_results = search_txt(
            question
        )

        if txt_results:

            with st.expander(
                "📄 View relevant TXT information"
            ):

                for result in txt_results:

                    st.caption(
                        f"{result['source']} "
                        f"| relevance: "
                        f"{result['score']:.2f}"
                    )

                    st.write(
                        result["text"][:1500]
                    )


    # -----------------------------------------
    # Save answer
    # -----------------------------------------

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# DATASET INFORMATION
# =========================================================

with st.expander(
    "📋 Knowledge Base Information"
):

    if st.session_state.csv_data:

        st.markdown("### 📊 CSV")

        for filename, df in st.session_state.csv_data.items():

            st.write(
                f"**{filename}** — "
                f"{len(df):,} rows × "
                f"{len(df.columns)} columns"
            )

            st.write(
                "Columns: "
                + ", ".join(
                    str(x)
                    for x in df.columns
                )
            )


    if st.session_state.txt_data:

        st.markdown("### 📄 TXT")

        for filename, text in st.session_state.txt_data.items():

            st.write(
                f"**{filename}** — "
                f"{len(text):,} characters"
            )
