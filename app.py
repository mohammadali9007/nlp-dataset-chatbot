import streamlit as st
import pandas as pd
import numpy as np
import re
import os

from google import genai


# =========================================================
# APP CONFIG
# =========================================================

st.set_page_config(
    page_title="IntelliMind AI - CSV Chatbot",
    page_icon="🤖",
    layout="wide"
)


# =========================================================
# CUSTOM STYLE
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
}

.app-title {
    font-size: 38px;
    font-weight: 700;
    margin-bottom: 5px;
}

.app-subtitle {
    color: #777;
    font-size: 16px;
    margin-bottom: 25px;
}

.info-box {
    padding: 18px;
    border-radius: 12px;
    background: rgba(128,128,128,0.08);
    border: 1px solid rgba(128,128,128,0.15);
}

.answer-box {
    padding: 20px;
    border-radius: 14px;
    background: rgba(70,130,180,0.08);
    border: 1px solid rgba(70,130,180,0.18);
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# TITLE
# =========================================================

st.markdown(
    '<div class="app-title">🤖 IntelliMind AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="app-subtitle">'
    'Ask questions about your CSV dataset using natural language.'
    '</div>',
    unsafe_allow_html=True
)


# =========================================================
# GEMINI API
# =========================================================

api_key = st.secrets.get("GEMINI_API_KEY", "")

if not api_key:
    api_key = os.getenv("GEMINI_API_KEY", "")

if api_key:
    try:
        client = genai.Client(api_key=api_key)
    except Exception:
        client = None
else:
    client = None


# =========================================================
# SESSION STATE
# =========================================================

if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("📁 Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(uploaded_file)

            st.session_state.df = df
            st.session_state.chat_history = []

            st.success("CSV loaded successfully!")

        except Exception as e:

            st.error(f"Could not read CSV: {e}")


    st.divider()

    if st.session_state.df is not None:

        df = st.session_state.df

        st.subheader("📊 Dataset Info")

        st.write(f"**Rows:** {df.shape[0]}")
        st.write(f"**Columns:** {df.shape[1]}")

        st.write("**Column names:**")

        for column in df.columns:
            st.write(f"• `{column}`")


    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):

        st.session_state.chat_history = []

        st.rerun()


# =========================================================
# CHECK DATASET
# =========================================================

if st.session_state.df is None:

    st.info(
        "👈 Upload a CSV file from the sidebar to start chatting."
    )

    st.markdown("### Example questions")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            **📊 General**
            
            • How many rows are there?
            
            • What are the columns?
            
            • Show me the dataset.
            """
        )

    with col2:
        st.markdown(
            """
            **🩺 Diabetes**
            
            • What is the average glucose?
            
            • How many diabetic patients?
            
            • Highest BMI?
            """
        )

    with col3:
        st.markdown(
            """
            **🔎 Analysis**
            
            • Find glucose above 200
            
            • Average age
            
            • Show missing values
            """
        )

    st.stop()


# =========================================================
# DATASET
# =========================================================

df = st.session_state.df


# =========================================================
# DATASET OVERVIEW
# =========================================================

st.subheader("📊 Dataset Overview")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Rows", df.shape[0])

with c2:
    st.metric("Columns", df.shape[1])

with c3:
    st.metric("Missing Values", int(df.isna().sum().sum()))

with c4:
    st.metric(
        "Numeric Columns",
        len(df.select_dtypes(include=np.number).columns)
    )


with st.expander("👀 Preview Dataset"):

    st.dataframe(
        df.head(20),
        use_container_width=True
    )


# =========================================================
# DATASET SUMMARY
# =========================================================

def create_dataset_summary(dataframe):

    summary = []

    summary.append(
        f"Dataset contains {dataframe.shape[0]} rows "
        f"and {dataframe.shape[1]} columns."
    )

    summary.append(
        "Columns: " + ", ".join(map(str, dataframe.columns))
    )

    summary.append("\nColumn information:")

    for col in dataframe.columns:

        dtype = str(dataframe[col].dtype)

        missing = int(dataframe[col].isna().sum())

        unique = int(dataframe[col].nunique())

        line = (
            f"- {col}: "
            f"type={dtype}, "
            f"unique_values={unique}, "
            f"missing={missing}"
        )

        if pd.api.types.is_numeric_dtype(dataframe[col]):

            line += (
                f", min={dataframe[col].min()}, "
                f"max={dataframe[col].max()}, "
                f"mean={dataframe[col].mean():.2f}"
            )

        summary.append(line)

    return "\n".join(summary)


dataset_summary = create_dataset_summary(df)


# =========================================================
# QUESTION PROCESSING
# =========================================================

def clean_question(question):

    question = question.strip()

    question = re.sub(
        r"\s+",
        " ",
        question
    )

    return question


# =========================================================
# FIND RELEVANT COLUMNS
# =========================================================

def find_relevant_columns(question, dataframe):

    question_lower = question.lower()

    relevant = []

    for column in dataframe.columns:

        col_lower = str(column).lower()

        words = re.findall(
            r"[a-zA-Z0-9]+",
            col_lower
        )

        for word in words:

            if len(word) >= 3 and word in question_lower:

                relevant.append(column)

                break

    # Diabetes-related common terms
    keyword_map = {

        "glucose": [
            "glucose",
            "sugar",
            "blood sugar"
        ],

        "age": [
            "age"
        ],

        "bmi": [
            "bmi",
            "body mass"
        ],

        "insulin": [
            "insulin"
        ],

        "pregnancy": [
            "pregnancy",
            "pregnancies"
        ],

        "blood pressure": [
            "blood pressure",
            "bp",
            "pressure"
        ],

        "diabetes": [
            "diabetes",
            "diabetic",
            "outcome"
        ]
    }

    for column in dataframe.columns:

        col_lower = str(column).lower()

        for concept, keywords in keyword_map.items():

            if any(k in question_lower for k in keywords):

                if (
                    concept in col_lower
                    or any(
                        k in col_lower
                        for k in keywords
                    )
                ):

                    if column not in relevant:
                        relevant.append(column)

    return relevant


# =========================================================
# CREATE RELEVANT DATA
# =========================================================

def get_relevant_data(question, dataframe):

    relevant_columns = find_relevant_columns(
        question,
        dataframe
    )

    # If specific columns found
    if relevant_columns:

        selected = dataframe[relevant_columns].copy()

        return selected, relevant_columns

    # Otherwise return numerical summary
    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns.tolist()

    if numeric_columns:

        return dataframe[numeric_columns].copy(), numeric_columns

    return dataframe.head(20).copy(), list(dataframe.columns)


# =========================================================
# STATISTICAL ANALYSIS
# =========================================================

def calculate_statistics(dataframe):

    stats = {}

    numeric_df = dataframe.select_dtypes(
        include=np.number
    )

    for column in numeric_df.columns:

        series = numeric_df[column].dropna()

        if len(series) == 0:
            continue

        stats[column] = {
            "count": int(series.count()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "min": float(series.min()),
            "max": float(series.max()),
            "std": float(series.std())
        }

    return stats


# =========================================================
# BASIC QUESTION ANSWER
# =========================================================

def basic_dataset_answer(question, dataframe):

    q = question.lower()

    # Number of rows
    if (
        "how many rows" in q
        or "number of rows" in q
        or "total rows" in q
    ):

        return (
            f"The dataset contains "
            f"**{len(dataframe):,} rows**."
        )

    # Number of columns
    if (
        "how many columns" in q
        or "number of columns" in q
    ):

        return (
            f"The dataset contains "
            f"**{len(dataframe.columns)} columns**."
        )

    # Column names
    if (
        "column names" in q
        or "what are the columns" in q
        or "list columns" in q
    ):

        columns = ", ".join(
            [f"`{c}`" for c in dataframe.columns]
        )

        return f"The columns are: {columns}"

    # Missing values
    if (
        "missing" in q
        or "null" in q
        or "empty" in q
    ):

        missing = dataframe.isna().sum()

        missing = missing[
            missing > 0
        ].sort_values(
            ascending=False
        )

        if len(missing) == 0:

            return "There are **no missing values** in the dataset."

        result = "\n".join(
            [
                f"- **{col}**: {int(value)} missing"
                for col, value in missing.items()
            ]
        )

        return (
            "The dataset contains these missing values:\n\n"
            + result
        )

    # Average questions
    if (
        "average" in q
        or "mean" in q
    ):

        relevant = find_relevant_columns(
            question,
            dataframe
        )

        numeric_columns = [
            col for col in relevant
            if pd.api.types.is_numeric_dtype(
                dataframe[col]
            )
        ]

        if numeric_columns:

            results = []

            for col in numeric_columns:

                value = dataframe[col].mean()

                results.append(
                    f"**{col}**: {value:.2f}"
                )

            return (
                "Average value:\n\n"
                + "\n".join(results)
            )

    # Maximum questions
    if (
        "highest" in q
        or "maximum" in q
        or "max" in q
        or "largest" in q
    ):

        relevant = find_relevant_columns(
            question,
            dataframe
        )

        numeric_columns = [
            col for col in relevant
            if pd.api.types.is_numeric_dtype(
                dataframe[col]
            )
        ]

        if numeric_columns:

            results = []

            for col in numeric_columns:

                value = dataframe[col].max()

                results.append(
                    f"**{col}**: {value:.2f}"
                )

            return (
                "Highest value:\n\n"
                + "\n".join(results)
            )

    # Minimum questions
    if (
        "lowest" in q
        or "minimum" in q
        or "min" in q
        or "smallest" in q
    ):

        relevant = find_relevant_columns(
            question,
            dataframe
        )

        numeric_columns = [
            col for col in relevant
            if pd.api.types.is_numeric_dtype(
                dataframe[col]
            )
        ]

        if numeric_columns:

            results = []

            for col in numeric_columns:

                value = dataframe[col].min()

                results.append(
                    f"**{col}**: {value:.2f}"
                )

            return (
                "Lowest value:\n\n"
                + "\n".join(results)
            )

    return None


# =========================================================
# GEMINI ANSWER
# =========================================================

def generate_ai_answer(
    question,
    dataframe,
    relevant_columns
):

    if client is None:

        return (
            "⚠️ Gemini API key is not configured.\n\n"
            "Please add `GEMINI_API_KEY` to Streamlit Secrets."
        )

    # Limit rows sent to AI
    preview_df = dataframe[
        relevant_columns
    ].head(100)

    csv_text = preview_df.to_csv(
        index=False
    )

    statistics = calculate_statistics(
        dataframe[relevant_columns]
    )

    statistics_text = str(statistics)

    prompt = f"""
You are IntelliMind AI, a data analysis chatbot.

The user uploaded a CSV dataset.

Your job is to answer the user's question using ONLY
the provided dataset information.

IMPORTANT:
- Do not invent data.
- Do not make up values.
- If the dataset does not contain enough information,
  clearly say that.
- Use simple English.
- Give a direct answer first.
- If useful, explain the calculation briefly.
- For medical/diabetes datasets, do not diagnose a person
  or provide treatment recommendations.
- You may describe statistics found in the dataset.
- If the question asks about a specific patient, only
  use information actually present in the dataset.

DATASET SUMMARY:
{dataset_summary}

STATISTICS:
{statistics_text}

RELEVANT CSV DATA:
{csv_text}

USER QUESTION:
{question}

Answer naturally and accurately.
"""

    try:

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text

    except Exception as e:

        return (
            "⚠️ AI generation error.\n\n"
            f"`{str(e)}`"
        )


# =========================================================
# CHAT HISTORY DISPLAY
# =========================================================

for message in st.session_state.chat_history:

    if message["role"] == "user":

        with st.chat_message("user"):
            st.write(message["content"])

    else:

        with st.chat_message("assistant"):
            st.markdown(message["content"])


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask something about your CSV..."
)


if question:

    question = clean_question(
        question
    )

    # User message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):
        st.write(question)


    # Find relevant data
    relevant_data, relevant_columns = get_relevant_data(
        question,
        df
    )


    # Try basic direct answer first
    direct_answer = basic_dataset_answer(
        question,
        df
    )


    if direct_answer:

        answer = direct_answer

    else:

        with st.spinner(
            "🔍 Analyzing your dataset..."
        ):

            answer = generate_ai_answer(
                question,
                df,
                relevant_columns
            )


    # Assistant message
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )


    with st.chat_message("assistant"):

        st.markdown(answer)

        # Show relevant data
        if len(relevant_data) > 0:

            with st.expander(
                "🔎 View relevant data"
            ):

                st.dataframe(
                    relevant_data.head(20),
                    use_container_width=True
                )


# =========================================================
# DATA ANALYSIS SECTION
# =========================================================

st.divider()

with st.expander("📈 Dataset Statistics"):

    numeric_df = df.select_dtypes(
        include=np.number
    )

    if len(numeric_df.columns) > 0:

        st.dataframe(
            numeric_df.describe().T,
            use_container_width=True
        )

    else:

        st.info(
            "No numerical columns found."
        )


# =========================================================
# DATA TYPES
# =========================================================

with st.expander("🔧 Column Information"):

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [
            str(dtype)
            for dtype in df.dtypes
        ],
        "Missing": [
            int(x)
            for x in df.isna().sum()
        ],
        "Unique Values": [
            int(x)
            for x in df.nunique()
        ]
    })

    st.dataframe(
        info_df,
        use_container_width=True,
        hide_index=True
    )
