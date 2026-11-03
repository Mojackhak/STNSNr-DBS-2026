if (any(!nzchar(Sys.getenv(c("MYLFP_PROJECT_ROOT"))))) {
  stop("Set MYLFP_PROJECT_ROOT before running this script.")
}

# --- Packages ---
suppressPackageStartupMessages({
  library(fs)
  library(readr)
  library(dplyr)
  library(rstudioapi)
})

DEFAULT_OUTPUT_ROOT <- file.path(Sys.getenv("MYLFP_PROJECT_ROOT"), "summary/stats/gait_cycle")
MIN_INTERVAL_SPAN <- 10.0
BASIS_CONFIGS <- list(
  lmer = list(
    basis = "lmer",
    active_col = "lmer_sig",
    direction_col = "lmer_direction",
    primary_estimate_col = "mean_estimate_lmer",
    output_dir = "lmer_model"
  ),
  rlmer = list(
    basis = "rlmer",
    active_col = "rlmer_sig",
    direction_col = "rlmer_direction",
    primary_estimate_col = "mean_estimate_rlmer",
    output_dir = "rlmer_model"
  ),
  joint = list(
    basis = "joint",
    active_col = "both_sig_same_direction",
    direction_col = "lmer_direction",
    primary_estimate_col = "mean_estimate_lmer",
    output_dir = "joint_model"
  )
)

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_flag <- "--file="
  file_arg <- args[startsWith(args, file_flag)]
  if (length(file_arg) > 0) {
    return(path_abs(path_dir(sub(file_flag, "", file_arg[[1]]))))
  }

  if (rstudioapi::isAvailable()) {
    editor_path <- rstudioapi::getSourceEditorContext()$path
    if (nzchar(editor_path)) {
      return(path_abs(path_dir(editor_path)))
    }
  }

  stop("Cannot determine script directory for summarize_intervals.R")
}

parse_cli_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  opts <- list(output_root = DEFAULT_OUTPUT_ROOT, deviation_csv = NULL)

  if (length(args) == 0) {
    return(opts)
  }

  i <- 1L
  while (i <= length(args)) {
    key <- args[[i]]
    if (key %in% c("--output-root", "--deviation-csv")) {
      if (i == length(args)) {
        stop("Missing value for ", key)
      }
      if (key == "--output-root") {
        opts$output_root <- args[[i + 1L]]
      } else {
        opts$deviation_csv <- args[[i + 1L]]
      }
      i <- i + 2L
      next
    }

    if (key %in% c("-h", "--help")) {
      cat(
        paste(
          "Usage: Rscript summarize_intervals.R [--output-root PATH]",
          "       Rscript summarize_intervals.R [--deviation-csv PATH]",
          "",
          "Defaults:",
          paste("  --output-root", DEFAULT_OUTPUT_ROOT),
          sep = "\n"
        )
      )
      quit(save = "no", status = 0)
    }

    stop("Unknown argument: ", key)
  }

  opts
}

write_output_table <- function(frame, csv_path) {
  dir_create(path_dir(csv_path), recurse = TRUE)
  write_csv(frame, csv_path)
}

safe_mean <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  mean(x, na.rm = TRUE)
}

safe_min <- function(x) {
  if (all(is.na(x))) {
    return(NA_real_)
  }
  min(x, na.rm = TRUE)
}

center_to_left_edge <- function(center_pct) {
  if (is.na(center_pct)) {
    return(NA_real_)
  }
  max(0.0, center_pct - 0.5)
}

center_to_right_edge <- function(center_pct, wraps_cycle = FALSE) {
  if (is.na(center_pct)) {
    return(NA_real_)
  }
  edge <- center_pct + 0.5
  if (wraps_cycle) {
    return(edge %% 100)
  }
  min(100.0, edge)
}

summarize_basis_segment <- function(
  data,
  basis_config,
  wraps_cycle = FALSE,
  start_pct = NULL,
  end_pct = NULL
) {
  if (is.null(start_pct)) {
    start_pct <- center_to_left_edge(data$t_pct[[1]])
  }
  if (is.null(end_pct)) {
    end_pct <- center_to_right_edge(data$t_pct[[nrow(data)]], wraps_cycle = wraps_cycle)
  }

  tibble::tibble(
    interval_basis = basis_config$basis,
    Metric = data$Metric[[1]],
    Band = data$Band[[1]],
    Region = data$Region[[1]],
    direction = data$interval_direction[[1]],
    start_pct = start_pct,
    end_pct = end_pct,
    span_pct = as.numeric(nrow(data)),
    wraps_cycle = wraps_cycle,
    n_points = nrow(data),
    mean_estimate_lmer = safe_mean(data$lmer_estimate),
    min_p_lmer = safe_min(data$lmer_p_value),
    mean_estimate_rlmer = safe_mean(data$rlmer_estimate),
    min_p_rlmer = safe_min(data$rlmer_p_value),
    robust_fraction = safe_mean(as.numeric(data$both_sig_same_direction))
  )
}

empty_interval_table <- function() {
  tibble::tibble(
    interval_basis = character(),
    Metric = character(),
    Band = character(),
    Region = character(),
    direction = character(),
    start_pct = numeric(),
    end_pct = numeric(),
    span_pct = numeric(),
    wraps_cycle = logical(),
    n_points = integer(),
    mean_estimate_lmer = numeric(),
    min_p_lmer = numeric(),
    mean_estimate_rlmer = numeric(),
    min_p_rlmer = numeric(),
    robust_fraction = numeric()
  )
}

build_param_intervals <- function(data, basis_config) {
  ordered <- data %>%
    arrange(t_pct) %>%
    mutate(
      interval_direction = .data[[basis_config$direction_col]],
      active = .data[[basis_config$active_col]] &
        (interval_direction %in% c("above_0", "below_0")),
      grid_index = row_number()
    )

  active <- ordered %>% filter(active)
  if (nrow(active) == 0L) {
    return(empty_interval_table())
  }

  break_flag <- c(
    TRUE,
    (active$interval_direction[-1] != active$interval_direction[-nrow(active)]) |
      ((active$grid_index[-1] - active$grid_index[-nrow(active)]) != 1L)
  )
  active$segment_id <- cumsum(break_flag)

  segments <- split(active, active$segment_id)
  merged_segments <- segments

  if (length(segments) >= 2L) {
    first_segment <- segments[[1]]
    last_segment <- segments[[length(segments)]]

    first_hits_zero <- first_segment$grid_index[[1]] == 1L
    last_hits_hundred <- last_segment$grid_index[[nrow(last_segment)]] == nrow(ordered)
    same_direction <- identical(
      first_segment$interval_direction[[1]],
      last_segment$interval_direction[[1]]
    )

    if (first_hits_zero && last_hits_hundred && same_direction) {
      wrap_segment <- bind_rows(last_segment, first_segment)
      wrap_summary <- summarize_basis_segment(
        wrap_segment,
        basis_config = basis_config,
        wraps_cycle = TRUE,
        start_pct = center_to_left_edge(last_segment$t_pct[[1]]),
        end_pct = center_to_right_edge(
          first_segment$t_pct[[nrow(first_segment)]],
          wraps_cycle = TRUE
        )
      )

      middle_segments <- if (length(segments) > 2L) {
        segments[2:(length(segments) - 1L)]
      } else {
        list()
      }

      merged_segments <- c(
        list(structure(list(data = wrap_segment, summary = wrap_summary), class = "wrapped_segment")),
        lapply(middle_segments, function(segment) {
          structure(
            list(data = segment, summary = summarize_basis_segment(segment, basis_config = basis_config)),
            class = "wrapped_segment"
          )
        })
      )
    } else {
      merged_segments <- lapply(segments, function(segment) {
        structure(
          list(data = segment, summary = summarize_basis_segment(segment, basis_config = basis_config)),
          class = "wrapped_segment"
        )
      })
    }
  } else {
    merged_segments <- lapply(segments, function(segment) {
      structure(
        list(data = segment, summary = summarize_basis_segment(segment, basis_config = basis_config)),
        class = "wrapped_segment"
      )
    })
  }

  bind_rows(lapply(merged_segments, `[[`, "summary")) %>%
    filter(span_pct >= MIN_INTERVAL_SPAN) %>%
    arrange(start_pct, end_pct)
}

build_candidate_table <- function(intervals_df, basis_config) {
  if (nrow(intervals_df) == 0L) {
    return(
      intervals_df %>%
        mutate(candidate_rank = integer()) %>%
        select(
          candidate_rank,
          interval_basis,
          Metric,
          Band,
          Region,
          direction,
          start_pct,
          end_pct,
          span_pct,
          wraps_cycle,
          n_points,
          mean_estimate_lmer,
          min_p_lmer,
          mean_estimate_rlmer,
          min_p_rlmer,
          robust_fraction
        )
    )
  }

  intervals_df %>%
    mutate(
      primary_abs_mean_estimate = abs(.data[[basis_config$primary_estimate_col]]),
      secondary_abs_mean_estimate = if (basis_config$basis == "joint") {
        abs(mean_estimate_rlmer)
      } else {
        NA_real_
      }
    ) %>%
    arrange(
      desc(span_pct),
      desc(primary_abs_mean_estimate),
      desc(secondary_abs_mean_estimate),
      desc(robust_fraction),
      Metric,
      Band,
      Region,
      direction
    ) %>%
    mutate(candidate_rank = row_number()) %>%
    select(
      candidate_rank,
      interval_basis,
      Metric,
      Band,
      Region,
      direction,
      start_pct,
      end_pct,
      span_pct,
      wraps_cycle,
      n_points,
      mean_estimate_lmer,
      min_p_lmer,
      mean_estimate_rlmer,
      min_p_rlmer,
      robust_fraction
    )
}

summarize_deviation_file <- function(deviation_csv) {
  band_dir <- path_dir(deviation_csv)
  message("Summarizing ", deviation_csv)

  deviation_df <- read_csv(
    deviation_csv,
    show_col_types = FALSE,
    col_types = cols(
      Metric = col_character(),
      Band = col_character(),
      Region = col_character(),
      t_pct = col_double(),
      lmer_estimate = col_double(),
      lmer_p_value = col_double(),
      lmer_status = col_character(),
      rlmer_estimate = col_double(),
      rlmer_p_value = col_double(),
      rlmer_status = col_character(),
      lmer_direction = col_character(),
      rlmer_direction = col_character(),
      lmer_sig = col_logical(),
      rlmer_sig = col_logical(),
      both_sig_same_direction = col_logical()
    )
  ) %>%
    arrange(Metric, Band, Region, t_pct)

  param_keys <- deviation_df %>%
    distinct(Metric, Band, Region) %>%
    arrange(Metric, Band, Region)

  for (basis_name in names(BASIS_CONFIGS)) {
    basis_config <- BASIS_CONFIGS[[basis_name]]
    interval_rows <- vector("list", length = nrow(param_keys))

    for (i in seq_len(nrow(param_keys))) {
      key <- param_keys[i, , drop = FALSE]
      data <- deviation_df %>%
        filter(
          Metric == key$Metric[[1]],
          Band == key$Band[[1]],
          Region == key$Region[[1]]
        )

      interval_rows[[i]] <- build_param_intervals(data, basis_config = basis_config)
    }

    intervals_df <- if (length(interval_rows) == 0L) {
      empty_interval_table()
    } else {
      bind_rows(interval_rows) %>%
        arrange(Metric, Band, Region, start_pct, end_pct)
    }

    candidates_df <- build_candidate_table(intervals_df, basis_config = basis_config)
    model_dir <- path(band_dir, basis_config$output_dir)
    write_output_table(intervals_df, path(model_dir, "timepoint_intervals.csv"))
    write_output_table(candidates_df, path(model_dir, "timepoint_interval_candidates.csv"))
  }
}

main <- function() {
  options <- parse_cli_args()
  output_root <- path_abs(options$output_root)

  if (!is.null(options$deviation_csv)) {
    deviation_csv <- path_abs(options$deviation_csv)
    if (!file_exists(deviation_csv)) {
      stop("Deviation CSV not found: ", deviation_csv)
    }
    summarize_deviation_file(deviation_csv)
    return(invisible(NULL))
  }

  if (!dir_exists(output_root)) {
    stop("Output root not found. Run export_region_long.py first: ", output_root)
  }

  deviation_files <- dir_ls(
    output_root,
    recurse = TRUE,
    type = "file",
    regexp = "timepoint_deviation\\.csv$"
  )
  deviation_files <- deviation_files[
    !path_file(path_dir(deviation_files)) %in% c("lmer_model", "rlmer_model", "joint_model")
  ]
  if (length(deviation_files) == 0L) {
    stop("No band-level timepoint_deviation.csv files found under ", output_root)
  }

  for (deviation_csv in deviation_files) {
    summarize_deviation_file(deviation_csv)
  }
}

SCRIPT_DIR <- get_script_dir()
main()
