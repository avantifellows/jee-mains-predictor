# =====================================================
# R Prediction Script for Streamlit (predict_in_R.R)
# =====================================================

# Get command-line arguments
args <- commandArgs(trailingOnly = TRUE)
input_path <- args[1]
output_path <- args[2]

# Load required packages
suppressMessages({
  library(readr)
  library(dplyr)
})

# Read input file
df <- read_csv(input_path, show_col_types = FALSE)

# Load trained models
reg_percentile <- readRDS("jee_percentile_model.rds")
reg_qualification <- readRDS("jee_qualification_model.rds")

# Run predictions
df$Predicted_JEE_percentile <- predict(reg_percentile, newdata = df)
df$Qualification_Probability <- predict(reg_qualification, newdata = df, type = "response")
df$Qualified_Prediction <- ifelse(df$Qualification_Probability >= 0.5, "Yes", "No")

# Save output file
write_csv(df, output_path)
