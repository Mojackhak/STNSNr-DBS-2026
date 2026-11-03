if (any(!nzchar(Sys.getenv(c("MYLFP_PROJECT_ROOT", "MYLFP_REPO_ROOT"))))) {
  stop("Set MYLFP_PROJECT_ROOT, MYLFP_REPO_ROOT before running this script.")
}

# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
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

source(file.path(Sys.getenv("MYLFP_REPO_ROOT"), "src/stats/emm/emm.R"))
source(file.path(Sys.getenv("MYLFP_REPO_ROOT"), "plugin/default/tabular_default.R"))
source(file.path(Sys.getenv("MYLFP_REPO_ROOT"), "src/tabular/df_schema.R"))

emm_options(pbkrtest.limit = 1e6, lmerTest.limit = 1e6)
emm_options(lmer.df = "kenward-roger")

# --- Set working directory to script location ---

project <- Sys.getenv("MYLFP_PROJECT_ROOT")
stats_dir <- path(project, "summary", "stats")
section_dir <- path(stats_dir, "rest")
params_dir_list <- c(path(section_dir, "conn"), 
                     path(section_dir, "gc"), 
                     path(section_dir, "psi"))
wk_dirs <- unlist(lapply(params_dir_list, list.dirs, recursive = TRUE, 
                         full.names = TRUE), use.names = FALSE)
wk_dirs <- wk_dirs[grepl("-scalar$", basename(wk_dirs))]

for (wk_dir in wk_dirs){
  
  # --- Load data ---
  df_path <- list.files(wk_dir, pattern = "contact_split\\.rds$",
                        recursive = FALSE, full.names = TRUE)[1]
  df <- readRDS(df_path)
  df <- normalize_df_categories(
    df = df,
    df_category = df_category,
  )
  
  df <- df %>%
    group_by(Band, Region) %>%
    mutate(
      zMX = scale(MNI_x_flip, center = TRUE, scale = TRUE)[,1],
      zMY = scale(MNI_y_flip, center = TRUE, scale = TRUE)[,1],
      zMZ = scale(MNI_z_flip, center = TRUE, scale = TRUE)[,1]
    ) %>%
    ungroup()
  
  # --- Run per Band ---
  Bands <- levels(df$Band)
  
  for (bd in Bands) {
    save_dir_bd <- path(wk_dir, bd)
    dir_create(save_dir_bd, recurse = TRUE)
    
    d <- df %>% filter(Band == bd) %>% droplevels()
    
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
      
      # --- Select Model ---      
      spec <- model_specs[[nm]]
      m <- do.call(
        spec$fn,
        c(list(formula = "Value ~ Region*(zMX + zMY + zMZ) + (1|ID)", data=d),
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
      
      
      # ---1.1 zMX-Region(β and p) ---
      base_name <- "zMX-Region"
      out <- test_numerical_var_effects(
        m,
        data = d,
        by = c("Region"),
        test_var = "zMX",
      )
      
      write_csv(out$omnibus, path(save_dir_nm, paste0(base_name, "_omnibus.csv")))
      write_csv(out$curve_df, path(save_dir_nm, paste0(base_name, "_curve.csv")))
      write_csv(out$slope_table, path(save_dir_nm, paste0(base_name, "_slope.csv")))    
      
      # ---1.2 zMY-Region(β and p) ---
      base_name <- "zMY-Region"
      out <- test_numerical_var_effects(
        m,
        data = d,
        by = c("Region"),
        test_var = "zMY",
      )
      
      write_csv(out$omnibus, path(save_dir_nm, paste0(base_name, "_omnibus.csv")))
      write_csv(out$curve_df, path(save_dir_nm, paste0(base_name, "_curve.csv")))
      write_csv(out$slope_table, path(save_dir_nm, paste0(base_name, "_slope.csv")))    
      
      # ---1.3 zMZ-Region(β and p) ---
      base_name <- "zMZ-Region"
      out <- test_numerical_var_effects(
        m,
        data = d,
        by = c("Region"),
        test_var = "zMZ",
      )
      
      write_csv(out$omnibus, path(save_dir_nm, paste0(base_name, "_omnibus.csv")))
      write_csv(out$curve_df, path(save_dir_nm, paste0(base_name, "_curve.csv")))
      write_csv(out$slope_table, path(save_dir_nm, paste0(base_name, "_slope.csv")))         
      
    }
  }
}



