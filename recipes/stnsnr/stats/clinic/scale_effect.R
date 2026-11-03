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
section_dir <- path(stats_dir, "clinic", "effect")
wk_dirs <- c(path(section_dir, "3m"),
             path(section_dir, "gait"),
             path(section_dir, "balance"))


for (wk_dir in wk_dirs){
  
  # --- Load data ---
  df_path <- list.files(wk_dir, pattern = "scale_subject\\.xlsx$",
                        recursive = FALSE, full.names = TRUE)[1]
  df <- read_excel(df_path)
  df <- na.omit(df)
  df <- normalize_df_categories(
    df = df,
    df_category = df_category,
  )
  df$Feature <- factor(df$Feature)
  
  # --- Run per Feature ---
  Features <- levels(df$Feature)
  
  for (bd in Features) {
    save_dir_bd <- path(wk_dir, bd)
    dir_create(save_dir_bd, recurse = TRUE)
    
    d <- df %>% filter(Feature == bd) %>% droplevels()
    
    if (nrow(d) <= 3) {
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
      
      save_dir_nm <- path(save_dir_bd, nm)
      dir_create(save_dir_nm, recurse = TRUE)
      
      if (nm == "lmer_model"){
        n_plots <- 1
      } else {
        n_plots <- 3
      }
      
      setwd(save_dir_nm)
      
      # --- Select Model ---      
      spec <- model_specs[[nm]]
      m <- do.call(
        spec$fn,
        c(list(formula = "Value ~ Group + (1|ID)", data=d),
          spec$args))  
      
      cairo_pdf("model_evaluation.pdf", width = 16, height = 16)
      print(performance::check_model(m)) 
      dev.off()
      
      base_name <- "model_evaluation"
      for (i in 1:n_plots) {
        cairo_pdf(paste0(base_name, as.character(i), ".pdf"), width = 8, height = 6)
        print(plot(m, which = i))
        dev.off()
      }     
      
      tab <- joint_tests(m)
      write_csv(tab, "model_omnibus.csv")
      
      cairo_pdf("model_evaluation.pdf", width = 16, height = 16)
      print(performance::check_model(m)) 
      dev.off()
      
      
      # ---1.1 Group ---
      base_name <- "Group"
      out <- EMM_single_effect(
        m = m,
        data = d,
        x_var = "Group",
      )
      
      write_csv(out$emm, path(save_dir_nm, paste0(base_name, "_emm.csv")))
      write_csv(out$tuk, path(save_dir_nm, paste0(base_name, "_tukey.csv")))

    }
  }
}



