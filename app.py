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
# SIMPLE PROFESSIONAL UI
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
}

.subtitle {
    color: #777;
    margin-bottom: 25px;
}

[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 15px;
}

</style>
""", unsafe_allow_html=True)


st.markdown(
    '<div class="title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'CSV + TXT Intelligent Knowledge Base Chatbot'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GEMINI API
# =========================================================

api_key = ""

try:
    api_key = st.secrets.get(
        "GEMINI_API_KEY",
        ""
    )
except Exception:
    pass

if not api_key:
    api_key = os.getenv(
        "GEMINI_API_KEY",
        ""
    )

client = None

if api_key:

    try:
        client = genai.Client(
            api_key=api_key
        )
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
# NORMALIZE TEXT
# =========================================================

def normalize_text(text):

    text = str(text).lower()

    text = text.replace("_", " ")
    text = text.replace("-", " ")

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
# FUZZY MATCH
# =========================================================

def fuzzy_match(
    word,
    choices,
    cutoff=0.65
):

    if not choices:
        return None

    word = normalize_text(word)

    normalized = {
        normalize_text(str(x)): x
        for x in choices
    }

    matches = difflib.get_close_matches(
        word,
        list(normalized.keys()),
        n=1,
        cutoff=cutoff
    )

    if matches:

        return normalized[
            matches[0]
        ]

    return None


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

        for uploaded_file in uploaded_files:

            filename = uploaded_file.name

            try:

                # =========================================
                # CSV
                # =========================================

                if filename.lower().endswith(".csv"):

                    df = pd.read_csv(
                        uploaded_file
                    )

                    st.session_state.csv_data[
                        filename
                    ] = df


                # =========================================
                # TXT
                # =========================================

                elif filename.lower().endswith(".txt"):

                    text = uploaded_file.read().decode(
                        "utf-8",
                        errors="ignore"
                    )

                    st.session_state.txt_data[
                        filename
                    ] = text

            except Exception as e:

                st.error(
                    f"Error reading {filename}: {e}"
                )


    st.divider()


    if st.session_state.csv_data:

        st.subheader("📊 CSV Files")

        for filename, df in st.session_state.csv_data.items():

            st.write(
                f"• {filename} "
                f"({len(df):,} rows)"
            )


    if st.session_state.txt_data:

        st.subheader("📄 TXT Files")

        for filename in st.session_state.txt_data:

            st.write(
                f"• {filename}"
            )


    st.divider()


    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# CHECK FILE
# =========================================================

if (
    not st.session_state.csv_data
    and
    not st.session_state.txt_data
):

    st.info(
        "👈 Upload CSV or TXT files to start."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.markdown("""
        ### 📊 CSV

        • Average glucose  
        • Highest BMI  
        • Number of patients  
        • Missing values
        """)

    with col2:

        st.markdown("""
        ### 📄 TXT

        • What is NLP?  
        • Explain AI  
        • Find information  
        • Summarize text
        """)

    with col3:

        st.markdown("""
        ### 🔀 Hybrid

        • CSV + TXT  
        • Typo tolerance  
        • Relevant answers  
        • Source display
        """)

    st.stop()


# =========================================================
# FILE METRICS
# =========================================================

total_csv = len(
    st.session_state.csv_data
)

total_txt = len(
    st.session_state.txt_data
)

total_rows = sum(
    len(df)
    for df in st.session_state.csv_data.values()
)


c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "📁 Total Files",
        total_csv + total_txt
    )

with c2:
    st.metric(
        "📊 CSV Files",
        total_csv
    )

with c3:
    st.metric(
        "📄 TXT Files",
        total_txt
    )


# =========================================================
# GET ALL COLUMNS
# =========================================================

def get_all_columns():

    columns = []

    for df in st.session_state.csv_data.values():

        for col in df.columns:

            if col not in columns:

                columns.append(col)

    return columns


# =========================================================
# GET NUMERIC COLUMNS
# =========================================================

def get_numeric_columns():

    columns = []

    for df in st.session_state.csv_data.values():

        numeric_columns = df.select_dtypes(
            include=np.number
        ).columns

        for col in numeric_columns:

            if col not in columns:

                columns.append(col)

    return columns


# =========================================================
# FIND RELEVANT CSV COLUMNS
# =========================================================

def find_relevant_columns(question):

    q = normalize_text(
        question
    )

    all_columns = get_all_columns()

    relevant = []


    # -----------------------------------------
    # Exact / partial match
    # -----------------------------------------

    for col in all_columns:

        normalized_col = normalize_text(
            col
        )

        if normalized_col in q:

            relevant.append(col)

            continue

        for word in normalized_col.split():

            if len(word) >= 3:

                if word in q:

                    relevant.append(col)

                    break


    # -----------------------------------------
    # Fuzzy matching
    # -----------------------------------------

    for word in q.split():

        if len(word) < 3:
            continue

        match = fuzzy_match(
            word,
            all_columns,
            cutoff=0.60
        )

        if match and match not in relevant:

            relevant.append(match)


    # -----------------------------------------
    # Diabetes synonyms
    # -----------------------------------------

    synonym_map = {

        "sugar": [
            "glucose"
        ],

        "blood sugar": [
            "glucose"
        ],

        "pressure": [
            "bloodpressure",
            "blood pressure"
        ],

        "pregnancy": [
            "pregnancies"
        ],

        "diabetes": [
            "outcome",
            "diabetes"
        ],

        "diabetic": [
            "outcome",
            "diabetes"
        ],

        "bmi": [
            "bmi"
        ],

        "insulin": [
            "insulin"
        ],

        "age": [
            "age"
        ]
    }


    for key, possible_columns in synonym_map.items():

        if key in q:

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
# SPLIT TXT INTO CHUNKS
# =========================================================

def split_text(
    text,
    chunk_size=1200
):

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


# =========================================================
# SEARCH TXT
# =========================================================

def search_txt(
    question,
    top_k=5
):

    documents = []

    sources = []


    for filename, text in st.session_state.txt_data.items():

        chunks = split_text(
            text
        )

        for chunk in chunks:

            documents.append(
                chunk
            )

            sources.append(
                filename
            )


    if not documents:

        return []


    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2)
        )

        matrix = vectorizer.fit_transform(
            documents
        )

        query_vector = vectorizer.transform(
            [question]
        )

        scores = cosine_similarity(
            query_vector,
            matrix
        )[0]

        ranked = np.argsort(
            scores
        )[::-1]


        results = []


        for index in ranked[:top_k]:

            if scores[index] > 0.02:

                results.append({

                    "text": documents[index],

                    "source": sources[index],

                    "score": float(
                        scores[index]
                    )
                })


        return results


    except Exception:

        return []


# =========================================================
# SEARCH CSV
# =========================================================

def search_csv(question):

    relevant_columns = find_relevant_columns(
        question
    )

    results = []


    for filename, df in st.session_state.csv_data.items():

        columns = [
            col
            for col in relevant_columns
            if col in df.columns
        ]


        if not columns:

            numeric = df.select_dtypes(
                include=np.number
            ).columns.tolist()

            columns = numeric[:10]


        if columns:

            results.append({

                "file": filename,

                "columns": columns,

                "data": df[columns].copy()

            })


    return results


# =========================================================
# DIRECT CSV ANALYSIS
# =========================================================

def direct_analysis(question):

    q = normalize_text(
        question
    )

    relevant_columns = find_relevant_columns(
        question
    )

    numeric_columns = get_numeric_columns()

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
            f"The dataset contains **{total:,} records**."
        )


    # =========================================
    # COLUMNS
    # =========================================

    if (
        "what are the columns" in q
        or "column names" in q
        or "list columns" in q
    ):

        answers.append(
            "The columns are:\n\n"
            +
            "\n".join(
                f"- `{col}`"
                for col in get_all_columns()
            )
        )


    # =========================================
    # AVERAGE
    # =========================================

    if (
        "average" in q
        or "mean" in q
        or "avg" in q
    ):

        target_columns = [
            col
            for col in relevant_columns
            if col in numeric_columns
        ]


        # Fuzzy numeric column detection
        if not target_columns:

            for word in q.split():

                match = fuzzy_match(
                    word,
                    numeric_columns,
                    cutoff=0.60
                )

                if match and match not in target_columns:

                    target_columns.append(
                        match
                    )


        for column in target_columns:

            values = []

            for df in st.session_state.csv_data.values():

                if column in df.columns:

                    vals = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    values.extend(
                        vals.tolist()
                    )


            if values:

                answers.append(
                    f"The average **{column}** is "
                    f"**{np.mean(values):.2f}**."
                )


    # =========================================
    # HIGHEST
    # =========================================

    if (
        "highest" in q
        or "maximum" in q
        or "largest" in q
        or "max value" in q
    ):

        target_columns = [
            col
            for col in relevant_columns
            if col in numeric_columns
        ]


        for column in target_columns:

            values = []

            for df in st.session_state.csv_data.values():

                if column in df.columns:

                    vals = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    values.extend(
                        vals.tolist()
                    )


            if values:

                answers.append(
                    f"The highest **{column}** is "
                    f"**{max(values):.2f}**."
                )


    # =========================================
    # LOWEST
    # =========================================

    if (
        "lowest" in q
        or "minimum" in q
        or "smallest" in q
        or "min value" in q
    ):

        target_columns = [
            col
            for col in relevant_columns
            if col in numeric_columns
        ]


        for column in target_columns:

            values = []

            for df in st.session_state.csv_data.values():

                if column in df.columns:

                    vals = pd.to_numeric(
                        df[column],
                        errors="coerce"
                    ).dropna()

                    values.extend(
                        vals.tolist()
                    )


            if values:

                answers.append(
                    f"The lowest **{column}** is "
                    f"**{min(values):.2f}**."
                )


    # =========================================
    # MISSING VALUES
    # =========================================

    if (
        "missing" in q
        or "null" in q
        or "empty" in q
    ):

        total_missing = 0

        for df in st.session_state.csv_data.values():

            total_missing += int(
                df.isna().sum().sum()
            )


        answers.append(
            f"The dataset contains "
            f"**{total_missing:,} missing values**."
        )


    # =========================================
    # UNIQUE
    # =========================================

    if (
        "unique" in q
        and relevant_columns
    ):

        for column in relevant_columns:

            for df in st.session_state.csv_data.values():

                if column in df.columns:

                    count = df[column].nunique()

                    answers.append(
                        f"`{column}` has "
                        f"**{count:,} unique values**."
                    )


    if answers:

        return "\n\n".join(
            dict.fromkeys(answers)
        )

    return None


# =========================================================
# CSV CONTEXT
# =========================================================

def build_csv_context(question):

    results = search_csv(
        question
    )

    if not results:

        return "No relevant CSV information found."


    context = []


    for result in results:

        data = result["data"].head(
            50
        )

        context.append(
            f"""
SOURCE FILE:
{result["file"]}

COLUMNS:
{", ".join(result["columns"])}

DATA:
{data.to_csv(index=False)}
"""
        )


    return "\n".join(
        context
    )


# =========================================================
# TXT CONTEXT
# =========================================================

def build_txt_context(question):

    results = search_txt(
        question
    )

    if not results:

        return "No relevant TXT information found."


    context = []


    for result in results:

        context.append(
            f"""
SOURCE FILE:
{result["source"]}

RELEVANCE SCORE:
{result["score"]:.3f}

TEXT:
{result["text"]}
"""
        )


    return "\n".join(
        context
    )


# =========================================================
# GEMINI
# =========================================================

def generate_ai_answer(
    question,
    csv_context,
    txt_context,
    direct_answer
):

    if client is None:

        if direct_answer:

            return direct_answer

        return (
            "⚠️ Gemini API key is missing.\n\n"
            "Add GEMINI_API_KEY to Streamlit Secrets."
        )


    prompt = f"""
You are IntelliMind AI.

You are a CSV and TXT knowledge-base assistant.

Answer the user's question using the uploaded
information.

IMPORTANT:

1. Use uploaded CSV/TXT information first.

2. Do not invent dataset values.

3. Understand small spelling mistakes.

Examples:

avarage -> average
glocose -> glucose
diabtes -> diabetes
patint -> patient

4. If a calculated result is provided,
trust that result.

5. If the files do not contain the answer,
say that clearly.

6. Keep the answer simple and direct.

7. If CSV and TXT both contain useful information,
combine them.

8. For diabetes or other medical datasets,
describe the data only.
Do not diagnose or prescribe treatment.

USER QUESTION:

{question}


CALCULATED CSV RESULT:

{direct_answer if direct_answer else "No direct calculation."}


CSV CONTEXT:

{csv_context}


TXT CONTEXT:

{txt_context}


Now give the most relevant answer.
"""


    try:

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )

        if response and response.text:

            return response.text

        return (
            "I could not generate an answer "
            "from the uploaded information."
        )


    except Exception as e:

        error = str(e)


        if "404" in error:

            return (
                "⚠️ Gemini model error.\n\n"
                f"Current model: `{MODEL_NAME}`\n\n"
                f"{error}"
            )


        if "429" in error:

            return (
                "⚠️ Gemini rate limit reached. "
                "Please try again later."
            )


        if (
            "401" in error
            or
            "403" in error
        ):

            return (
                "⚠️ Gemini API key problem.\n\n"
                "Check GEMINI_API_KEY."
            )


        return (
            "⚠️ Gemini API Error:\n\n"
            f"{error}"
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
    "Ask anything about your files..."
)


if question:

    # =========================================
    # USER
    # =========================================

    st.session_state.chat_history.append({
        "role": "user",
        "content": question
    })


    with st.chat_message("user"):

        st.markdown(
            question
        )


    # =========================================
    # ASSISTANT
    # =========================================

    with st.chat_message("assistant"):

        with st.spinner(
            "🔎 Searching your files..."
        ):

            direct_answer = direct_analysis(
                question
            )

            csv_context = build_csv_context(
                question
            )

            txt_context = build_txt_context(
                question
            )


        with st.spinner(
            "🤖 Generating answer..."
        ):

            answer = generate_ai_answer(
                question,
                csv_context,
                txt_context,
                direct_answer
            )


        st.markdown(
            answer
        )


        # =====================================
        # CSV SOURCE
        # =====================================

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


        # =====================================
        # TXT SOURCE
        # =====================================

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
                        f"| score: "
                        f"{result['score']:.2f}"
                    )

                    st.write(
                        result["text"][:1500]
                    )


    # =========================================
    # SAVE ASSISTANT
    # =========================================

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer
    })


# =========================================================
# KNOWLEDGE BASE INFO
# =========================================================

with st.expander(
    "📋 Knowledge Base Information"
):

    if st.session_state.csv_data:

        st.markdown(
            "### 📊 CSV Files"
        )

        for filename, df in st.session_state.csv_data.items():

            st.write(
                f"**{filename}**"
            )

            st.write(
                f"Rows: {len(df):,} | "
                f"Columns: {len(df.columns)}"
            )

            st.write(
                "Columns: "
                +
                ", ".join(
                    str(c)
                    for c in df.columns
                )
            )


    if st.session_state.txt_data:

        st.markdown(
            "### 📄 TXT Files"
        )

        for filename, text in st.session_state.txt_data.items():

            st.write(
                f"**{filename}**"
            )

            st.write(
                f"Characters: {len(text):,}"
            )


# =========================================================
# CSV PREVIEW
# =========================================================

with st.expander(
    "👀 Preview CSV Data"
):

    for filename, df in st.session_state.csv_data.items():

        st.markdown(
            f"### {filename}"
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )
