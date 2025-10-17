# install_packages.R

# 1. Define a writable path within the user's home directory
# Streamlit Cloud's deployment user is typically allowed to write to their home directory (~).
user_library_path <- Sys.getenv("R_LIBS_USER", unset = "~/R/x86_64-pc-linux-gnu-library/4.3")

# 2. Create the directory if it doesn't exist
dir.create(user_library_path, recursive = TRUE, showWarnings = FALSE)

# 3. Set the default library path for the session
.libPaths(user_library_path)

# 4. Install the package using the new, writable path
install.packages("readr", repos = "https://cloud.r-project.org/")
install.packages("dplyr", repos = "https://cloud.r-project.org/")

