import streamlit as st
import pandas as pd
import subprocess
import tempfile
import os


def preprocess_input_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and prepares user-uploaded dataset for R model predictions.
    Raises errors if required columns are missing or contain unexpected values.
    """

    df = df.copy()

    # --- Standardize column names ---
    df.columns = [col.strip().lower() for col in df.columns]

    # --- Required columns ---
    required_cols = ["gender", "category", "program", "tenth_math_final", 
                     "tenth_sci_final", "pcm", "jeecutoff"]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # --- Handle Gender ---
    df["gender"] = df["gender"].str.strip().str.lower()
    allowed_genders = ["male", "female"]
    invalid_genders = df.loc[~df["gender"].isin(allowed_genders), "gender"].unique()
    if len(invalid_genders) > 0:
        raise ValueError(f"Invalid Gender values found: {', '.join(invalid_genders)}")
    df["female"] = df["gender"].apply(lambda x: 1 if x == "female" else 0)

    # --- Handle Category ---
    df["category"] = df["category"].str.strip().str.lower()
    allowed_categories = ["general", "obc", "sc", "st"]
    invalid_categories = df.loc[~df["category"].isin(allowed_categories), "category"].unique()
    if len(invalid_categories) > 0:
        raise ValueError(f"Invalid Category values found: {', '.join(invalid_categories)}")
    
    for cat in allowed_categories:
        df[cat] = df["category"].apply(lambda x: 1 if x == cat else 0)

    # --- Handle Program ---
    df["program"] = df["program"].str.strip().str.lower()
    program_dummies = {
        "avanti_coe": ["avanti coe"],
        "avanti_nodal": ["avanti nodal"],
        "enf_coe": ["enf coe"],
        "dakshana_coe": ["dakshana coe"],
        "jnv_enable": ["enable"]
    }

    allowed_programs = sum(program_dummies.values(), [])
    invalid_programs = df.loc[~df["program"].isin(allowed_programs), "program"].unique()
    if len(invalid_programs) > 0:
        raise ValueError(f"Invalid Program values found: {', '.join(invalid_programs)}")

    for col, keywords in program_dummies.items():
        df[col] = df["program"].apply(lambda x: 1 if any(k in x for k in keywords) else 0)

    # --- Numeric columns ---
    numeric_cols = ["tenth_math_final", "tenth_sci_final", "pcm", "jeecutoff"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().any():
            raise ValueError(f"Non-numeric values found in column: {col}")
        df[col] = df[col].fillna(0)

     # --- Final column order (model input) ---
    model_cols = [
        "avanti_coe", "avanti_nodal", "enf_coe", "dakshana_coe",
        "jnv_enable", "female", "tenth_math_final", "tenth_sci_final",
        "pcm", "general", "obc", "sc", "st", "jeecutoff"
    ]

    final_df = pd.concat([df[[col for col in df.columns if col not in model_cols]],
                          df[model_cols]], axis=1)

    return final_df




#Setting up the Streamlit UI

st.set_page_config(page_title="JEE Predictor", layout="wide")
st.title("🎯 JEE Percentile & Qualification Predictor")

st.markdown("""  
The app will use **Regression models** to generate:
- Predicted JEE Percentile  
- Qualification Probability  
- Qualification Status (Yes/No) using qualification probability (Student is predicted to qualify if Qualification Probability is above 0.5)
""")

with st.expander("📊 Model Accuracy Summary"):
    st.markdown("""

> ⚡ Note: These metrics provide an overview of the models' performance.  
> - **Accuracy** shows overall correct predictions.  
> - **Sensitivity** measures true positives.  
> - **Specificity** measures true negatives.  
> - **Precision** is the proportion of predicted positives that are true positives.  
> - **F1 Score** balances precision and sensitivity.                
                
                
### Predicted Qualification (Probability > 0.6)
| Metric | Regression (Logit) |
|--------|------------------|
| **Accuracy** | 0.80 |
| **Sensitivity** | 0.42 |
| **Specificity** | 0.95 |
| **Precision** | 0.78 |
| **F1 Score** | 0.54 |

### Qualification Based on Predicted Percentiles
| Metric | Regression |
|--------|-----------|
| **Accuracy** | 0.81 |
| **Sensitivity** | 0.49 |
| **Specificity** | 0.93 |
| **Precision** | 0.74 |
| **F1 Score** | 0.59 |

""")



with st.expander("📘 Data Upload Instructions (Click to View)"):
    st.markdown("""
### 🧩 1. File Format
- Upload your data as **CSV** or **Excel (.xlsx)**.
- Each row should represent **one student**.
- Column names should be clear and consistent.

---

### 📋 2. Required Columns (Make sure the column names are as given below)

| **Column Name** | **Type / Format** | **Description** | **Example** |
|-----------------|-----------------|-----------------|-------------|
| `StudentName` *(optional)* | Text | Student’s name (for identification only) | `Rahul Kumar` |
| `Student ID` | Numeric | Student’s ID (for identification only) | `123456789` |
| `Gender` | Text | Enter **Male** or **Female** | `Male` |
| `Category` | Text | One of: **General**, **OBC**, **SC**, **ST** | `OBC` |
| `Program` | Text | One of: **Avanti COE**, **Avanti Nodal**, **ENF COE**, **Dakshana COE**, **Enable** | `Avanti COE` |
| `tenth_math_final` | Numeric | Class 10 Math percentage | `85.5` |
| `tenth_sci_final` | Numeric | Class 10 Science percentage | `89` |
| `pcm` | Numeric | 1 if student took PCM in 11th, 0 if student took PCMB | `1` |
| `jeeCutoff` | Numeric | JEE 2024 or 2025 qualifying cutoff percentile | `90.778` |

---

### 🛠 3. Automatic Columns Created by the System
The app automatically generates the following dummy columns:

| **New Column** | **Created From** | **Description** |
|----------------|-----------------|-----------------|
| `female` | `Gender` | 1 if Female, else 0 |
| `general`, `obc`, `sc`, `st` | `Category` | Category dummy variables |
| `avanti_coe`, `avanti_nodal`, `enf_coe`, `dakshana_coe` | `Program` | Program dummy variables |

These are used internally by the models.

---

### ⚠️ 4. Data Shape Requirements
- Minimum **1 student record**, multiple rows allowed.
- Missing numeric values will be treated as **0**.
- Text columns (`Gender`, `Category`, `Program`) are **case-insensitive**. Example: `male`, `Male`, and `MALE` are all accepted.

---

### 🎯 5. Output
- After processing, the app will return the uploaded file with **predictions**:
  - `Model1_Prediction`
  - `Model2_Prediction`
- You can **download the results as CSV** for further use.

---

### 💡 Example Input (simplified)

| StudentName | Gender | Category | Program | tenth_math_final | tenth_sci_final | pcm | jeeCutoff |
|-------------|--------|----------|---------|------------------|-----------------|-----|-----------|
| Riya Sharma | Female | OBC | Avanti COE  | 88 | 91 | 1 | 90.778 |
| Aman Gupta | Male | General | ENF COE | 92 | 95 | 0 | 90.778 |

---
""")


uploaded_file = st.file_uploader("📂 Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read uploaded file
        if uploaded_file.name.endswith(".csv"):
            df_raw = pd.read_csv(uploaded_file)
        else:
            df_raw = pd.read_excel(uploaded_file)

        st.write("📋 Raw data preview:")
        st.dataframe(df_raw.head())

        # Preprocess uploaded data
        df = preprocess_input_data(df_raw)

        st.write("✅ Data prepared for model input:")
        st.dataframe(df.head())



        # Run predictions
        if st.button("🔮 Run Predictions"):
            with st.spinner("Running R model predictions... ⏳"):
                # Save input temp file
                input_path = tempfile.mktemp(suffix=".csv")
                output_path = tempfile.mktemp(suffix=".csv")
                df.to_csv(input_path, index=False)

                # Run the R script
                result = subprocess.run(
                    ["Rscript", "predict_in_R.R", input_path, output_path],
                    capture_output=True,
                    text=True
                )

                if result.returncode != 0:
                    st.error("⚠️ R script failed:")
                    st.code(result.stderr)
                else:
                    pred_df = pd.read_csv(output_path)
                    st.success("✅ Predictions completed!")
                    st.dataframe(pred_df.head())

                    csv_data = pred_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Results as CSV",
                        csv_data,
                        "jee_predictions.csv",
                        "text/csv"
                    )

                    # Optional: clean up temporary files
                    os.remove(input_path)
                    os.remove(output_path)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.info("👆 Upload a file to start predictions.")
    




