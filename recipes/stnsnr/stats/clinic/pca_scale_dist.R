if (any(!nzchar(Sys.getenv(c("MYLFP_PROJECT_ROOT", "MYLFP_REPO_ROOT"))))) {
  stop("Set MYLFP_PROJECT_ROOT, MYLFP_REPO_ROOT before running this script.")
}

# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
  library(readxl)
  library(lme4)
  library(lmerTest)
  library(glmmTMB)
  library(gamm4)
  library(mgcv)
  library(emmeans)
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(broom.mixed)
  library(readr)
  library(stringr)
  library(effects)
  library(sjPlot)
  library(ggeffects)
  library(ggpubr)
  library(ggplot2)
  library(ggbreak)
  library(rlang)
  library(robustlmm)
  library(rstudioapi)
  library(Cairo)
  library(performance)
  library(DHARMa)
  library(influence.ME)
  library(splines)
})

repo_dir <- Sys.getenv("MYLFP_REPO_ROOT")
source(path(repo_dir, "src", "stats", "emm", "emm.R"))
source(path(repo_dir, "plugin", "default", "tabular_default.R"))
source(path(repo_dir, "src", "tabular", "df_schema.R"))

emm_options(pbkrtest.limit = 1e6, lmerTest.limit = 1e6)
emm_options(lmer.df = "kenward-roger")
# --- Set working directory to script location ---

project <- Sys.getenv("MYLFP_PROJECT_ROOT")
stats_dir <- path(project, "summary", "stats")
section_dir <- path(stats_dir, "clinic", "pca")
wk_dirs <- c(path(section_dir, "3m"))


for (wk_dir in wk_dirs){
  
  # --- Load data ---
  df_path <- list.files(wk_dir, pattern = "dist\\.xlsx$",
                        recursive = FALSE, full.names = TRUE)[1]
  df <- read_excel(df_path)
  df <- na.omit(df)
  df <- normalize_df_categories(
    df = df,
    df_category = df_category,
  )
  
  if (nrow(df) <= 3) {
    next
  }
  
  model_specs <- list(
    lmer_model = list(
      fn   = lmer,
      args = list(
        REML = TRUE)
    ),
    rlmer_model = list(
      fn   = rlmer,
      args = list(
        method = "DAStau")
    )
  )       
  
  for (nm in names(model_specs)) {
    
    save_dir_nm <- path(wk_dir, nm)
    dir_create(save_dir_nm, recurse = TRUE)
    
    if (nm == "lmer_model"){
      n_plots <- 1
    } else {
      n_plots <- 3
    }
    
    # --- Select Model ---      
    spec <- model_specs[[nm]]
    m <- do.call(
      spec$fn,
      c(list(formula = "Value ~ Protocol + (1|ID)", data=df),
        spec$args))  
    
    tab <- joint_tests(m)
    write_csv(tab, path(save_dir_nm, "model_omnibus.csv"))
    
    cairo_pdf(path(save_dir_nm, "model_evaluation.pdf"), width = 16, height = 16)
    print(performance::check_model(m)) 
    dev.off()
    
    base_name <- "model_evaluation"
    for (i in 1:n_plots) {
      cairo_pdf(path(save_dir_nm, paste0(base_name, as.character(i), ".pdf")), width = 8, height = 6)
      print(plot(m, which = i))
      dev.off()
    }
    
    
    ############## RAW ##############
    
    # ---1.1. Protocol ---
    base_name <- "Protocol"
    out <- EMM_single_effect(
      m,
      data = df,
      x_var = "Protocol")
    
    write_csv(out$emm, path(save_dir_nm, paste0(base_name, "_emm.csv")))
    write_csv(out$tuk, path(save_dir_nm, paste0(base_name, "_tukey.csv")))
    
    
  }
}


