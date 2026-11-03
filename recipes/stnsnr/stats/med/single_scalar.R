if (any(!nzchar(Sys.getenv(c("MYLFP_PROJECT_ROOT", "MYLFP_REPO_ROOT"))))) {
  stop("Set MYLFP_PROJECT_ROOT, MYLFP_REPO_ROOT before running this script.")
}

# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
  library(lme4)
  library(lmerTest)
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
section_dir <- path(stats_dir, "med")
params_dir_list <- c(path(section_dir, "tfr"), 
                     path(section_dir, "params"), 
                     path(section_dir, "burst"))
wk_dirs <- unlist(lapply(params_dir_list, list.dirs, recursive = TRUE, 
                         full.names = TRUE), use.names = FALSE)
wk_dirs <- wk_dirs[grepl("-scalar$", basename(wk_dirs))]

for (wk_dir in wk_dirs){
  
  # --- Load data ---
  df_path <- list.files(wk_dir, pattern = "region_raw\\.rds$",
                        recursive = FALSE, full.names = TRUE)[1]
  df <- readRDS(df_path)
  df <- normalize_df_categories(
    df = df,
    df_category = df_category,
  )
  
  df_norm_path <- list.files(wk_dir, pattern = "region_normalized\\.rds$",
                             recursive = FALSE, full.names = TRUE)[1]
  df_norm <- readRDS(df_norm_path)
  df_norm <- normalize_df_categories(
    df = df_norm,
    df_category = df_category,
  )  
  
  # --- Run per Band ---
  Bands <- levels(df$Band)
  
  for (bd in Bands) {
    save_dir_bd <- path(wk_dir, bd)
    dir_create(save_dir_bd, recurse = TRUE)
    
    d <- df %>% filter(Band == bd) %>% droplevels()
    d_norm <- df_norm %>% filter(Band == bd) %>% droplevels()
    
    if (min(nrow(d), nrow(d_norm)) <= 3) {
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
      
      # --- Select Model ---      
      spec <- model_specs[[nm]]
      m <- do.call(
        spec$fn,
        c(list(formula = "Value ~ Med*Region + (1|ID)", data=d),
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
      
      
      m_norm <- do.call(
        spec$fn,
        c(list(formula = "Value ~ Region + (1|ID)", data=d_norm),
          spec$args))  
      
      tab_norm <- joint_tests(m_norm)
      write_csv(tab_norm, path(save_dir_nm, "model_omnibus_norm.csv"))
      
      cairo_pdf(path(save_dir_nm, "model_evaluation_norm.pdf"), width = 16, height = 16)
      print(performance::check_model(m_norm)) 
      dev.off()
      
      base_name_norm <- "model_evaluation_norm"
      for (i in 1:n_plots) {
        cairo_pdf(path(save_dir_nm, paste0(base_name_norm, as.character(i), ".pdf")), width = 8, height = 6)
        print(plot(m_norm, which = i))
        dev.off()
      }
      
      ############## RAW ##############
      
      # ---1.1. Med-Region ---
      base_name <- "Med-Region"
      out <- EMM_double_interaction(
        m = m,
        data = d,
        x_var = "Med",
        panel_var = "Region",
      )
      
      write_csv(out$emm, path(save_dir_nm, paste0(base_name, "_emm.csv")))
      write_csv(out$tuk, path(save_dir_nm, paste0(base_name, "_tukey.csv")))
      
      ############## NORM ##############
      
      # ---2.1. Region ---
      base_name <- "Region"
      out <- EMM_single_effect(
        m = m_norm,
        data = d_norm,
        x_var = "Region",
      )
      
      write_csv(out$emm, path(save_dir_nm, paste0(base_name, "_emm.csv")))
      write_csv(out$tuk, path(save_dir_nm, paste0(base_name, "_tukey.csv")))
      
      
    }
  }
}

