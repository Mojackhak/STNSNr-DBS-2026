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

repo_dir <- Sys.getenv("MYLFP_REPO_ROOT")
source(path(repo_dir, "src", "stats", "emm", "emm.R"))
source(path(repo_dir, "plugin", "default", "tabular_default.R"))
source(path(repo_dir, "src", "tabular", "df_schema.R"))

emm_options(pbkrtest.limit = 1e6, lmerTest.limit = 1e6)
emm_options(lmer.df = "kenward-roger")

# --- Set working directory to script location ---

project <- Sys.getenv("MYLFP_PROJECT_ROOT")
stats_dir <- path(project, "summary", "stats")
section_list_dir <- file.path(stats_dir, c("turn_feature", "turn_feature_percent"))
params <- c("conn", "gc", "psi")
grid <- expand.grid(section_dir = section_list_dir, param = params, stringsAsFactors = FALSE)
params_dir_list <- with(grid, file.path(section_dir, param))
wk_dirs <- unlist(lapply(params_dir_list, list.dirs, recursive = TRUE, 
                         full.names = TRUE), use.names = FALSE)
wk_dirs <- wk_dirs[grepl("-scalar$", basename(wk_dirs))]

for (wk_dir in wk_dirs){
  
  # --- Load data ---
  df_path <- list.files(wk_dir, pattern = "region_normalized\\.rds$",
                        recursive = FALSE, full.names = TRUE)[1]
  df <- readRDS(df_path)
  df <- normalize_df_categories(
    df = df,
    df_category = df_category,
  )
  
  Features <- unique(df$Feature)
  
  df <- df %>%
    group_by(Region, Feature, Band, ID) %>%
    mutate(
      feature_mean_id = mean(Value_feature, na.rm = TRUE),
      feature_within  = Value_feature - feature_mean_id
    ) %>%
    ungroup() %>%
    group_by(Region, Band, Feature) %>%
    mutate(
      feature_grand_mean = mean(Value_feature, na.rm = TRUE),
      feature_between    = feature_mean_id - feature_grand_mean,
      z_Feature_within   = as.numeric(scale(feature_within)),
      z_Feature_between  = as.numeric(scale(feature_between))
    ) %>%
    ungroup()
  
  for (ft in Features){
    df_feature <- df %>% filter(Feature == ft) %>% droplevels()
    
    # --- Run per Band ---
    Bands <- levels(df_feature$Band)
    
    for (bd in Bands) {
      save_dir_bd <- path(wk_dir, bd)
      dir_create(save_dir_bd, recurse = TRUE)
      
      d <- df_feature %>% filter(Band == bd) %>% droplevels()
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
          c(list(formula = "Value ~ z_Feature_within + z_Feature_between + (1|ID)", data=d),
            spec$args))  
        
        tab <- joint_tests(m)
        write_csv(tab, path(save_dir_nm, paste0("model_omnibus_", ft, ".csv")))
        
        cairo_pdf(path(save_dir_nm, paste0("model_evaluation_", ft, ".pdf")), width = 16, height = 16)
        print(performance::check_model(m)) 
        dev.off()
        
        base_name <- "model_evaluation"
        for (i in 1:n_plots) {
          cairo_pdf(path(save_dir_nm, paste0(base_name, "_", ft, as.character(i), ".pdf")), width = 8, height = 6)
          print(plot(m, which = i))
          dev.off()
        }
        
        
        # ---1.1 z_Feature_within(β and p) ---
        base_name <- "z_Feature_within"
        out <- test_numerical_var_effects(
          m,
          data = d,
          by = c(NULL),
          test_var = "z_Feature_within",
        )
        
        write_csv(out$omnibus, path(save_dir_nm, paste0(base_name,"_", ft, "_omnibus.csv")))
        write_csv(out$curve_df, path(save_dir_nm, paste0(base_name, "_", ft, "_curve.csv")))
        write_csv(out$slope_table, path(save_dir_nm, paste0(base_name, "_", ft,"_slope.csv")))  
        
        
        # ---1.1 z_Feature_between(β and p) ---
        base_name <- "z_Feature_between"
        out <- test_numerical_var_effects(
          m,
          data = d,
          by = c(NULL),
          test_var = "z_Feature_between",
        )
        
        write_csv(out$omnibus, path(save_dir_nm, paste0(base_name,"_", ft, "_omnibus.csv")))
        write_csv(out$curve_df, path(save_dir_nm, paste0(base_name, "_", ft, "_curve.csv")))
        write_csv(out$slope_table, path(save_dir_nm, paste0(base_name, "_", ft,"_slope.csv")))  
        
      }  
    }
  }
}

