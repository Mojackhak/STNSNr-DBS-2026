#' Normalize data.frame columns to factors based on a category spec.
#'
#' For each (column -> categories) in df_category:
#' - If the column exists in df, coerce it to a factor with the given levels.
#' - mode = "drop": values not in the declared categories become NA.
#' - mode = "append": values not in the declared categories are preserved and appended
#'   to the end of the factor levels in order of appearance.
#'
#' Then:
#' - Optionally drop rows with NA in any normalized category column.
#' - Drop unused factor levels (droplevels).
#' - Reset row names.
#'
#' @param df A data.frame.
#' @param df_category A named list: names are column names, each value is a vector of categories.
#' @param ordered Logical; whether created factors are ordered. Default TRUE.
#' @param mode Character; "drop" or "append". Default "drop".
#' @param dropna Logical; if TRUE, drop rows with NA in any normalized category column. Default TRUE.
#'
#' @return A normalized data.frame.
normalize_df_categories <- function(df,
                                    df_category,
                                    ordered = TRUE,
                                    mode = c("append", "drop"),
                                    dropna = TRUE) {
  mode <- match.arg(mode)
  
  if (!is.data.frame(df)) {
    stop(sprintf("df must be a data.frame, got %s", class(df)[1]))
  }
  if (!is.list(df_category) || is.null(names(df_category))) {
    stop("df_category must be a named list, e.g. list(Region=c('SNr','STN'))")
  }
  
  out <- df
  normalized_cols <- character(0)
  
  unique_preserve_order <- function(x) {
    x_chr <- as.character(x)
    x_chr[!duplicated(x_chr)]
  }
  
  for (col in names(df_category)) {
    if (!col %in% names(out)) next
    
    base_levels <- unique_preserve_order(df_category[[col]])
    
    x_chr <- as.character(out[[col]]) # keep NA as NA
    
    if (mode == "append") {
      observed <- unique(x_chr[!is.na(x_chr)]) # appearance order
      extras <- observed[!(observed %in% base_levels)]
      levels <- c(base_levels, extras)
    } else {
      levels <- base_levels
    }
    
    out[[col]] <- factor(x_chr, levels = levels, ordered = ordered)
    normalized_cols <- c(normalized_cols, col)
  }
  
  if (dropna && length(normalized_cols) > 0) {
    keep <- complete.cases(out[, normalized_cols, drop = FALSE])
    out <- out[keep, , drop = FALSE]
  }
  
  out <- droplevels(out)
  row.names(out) <- NULL
  out
}
