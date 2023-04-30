#### Nevergrad installation guide
## ATTENTION: Python version 3.10+ version may cause Nevergrad error
## See here for more info about installing Python packages via reticulate
## https://rstudio.github.io/reticulate/articles/python_packages.html

# Install reticulate first if you haven't already
install.packages("reticulate")

#### Option 2: nevergrad installation via conda
# 1. load reticulate
library("reticulate")
# 2. Install conda if not available
install_miniconda()
# 3. create virtual environment
conda_create("r-reticulate")
# 4. use the environment created
use_condaenv("r-reticulate")
# 5. point Python path to the python file in the virtual environment. Below is
#    an example for MacOS M1 or above. The "~" is my home dir "/Users/gufengzhou".
#    Show hidden files in case you want to locate the file yourself
Sys.setenv(RETICULATE_PYTHON = "~/Library/r-miniconda/envs/r-reticulate/bin/python")
# Sys.setenv(RETICULATE_PYTHON = "~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python")
# 6. Check python path
py_config() # If the first path is not as 5, do 7
# 7. Restart R session, run #5 first, then load library("reticulate"), check
#    py_config() again, python should have path as in #5
# 8. Install numpy if py_config shows it's not available
conda_install("r-reticulate", c("numpy<1.23.5"), pip=TRUE)
# 9. Install nevergrad
conda_install("r-reticulate", "nevergrad", pip=TRUE)
# 10. Install lingam
conda_install("r-reticulate", "lingam", pip=TRUE)
# 11. Install pandas
conda_install("r-reticulate", "pandas", pip=TRUE)
# 12. Install graphviz
conda_install("r-reticulate", "graphviz", pip=TRUE)
# 13. Install sklearn
conda_install("r-reticulate", "sklearn", pip=TRUE)

# 14. If successful, py_config() should show numpy and nevergrad with installed paths
# 15. Everytime R session is restarted, you need to run #4 first to assign python
#    path before loading Robyn
# 16. Alternatively, add the line RETICULATE_PYTHON = "~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python"
#    in the file Renviron in the the R directory to force R to always use this path by
#    default. Run R.home() to check R path, then look for file "Renviron". This way, you
#    don't need to run #5 everytime. Note that forcing this path impacts other R instances
#    that requires different Python versions