# Numerical-predictor effects used by coordinate mixed-model analyses.

.lfp_stats_adjust_numerical_table <- function(table, method, by = NULL) {
  if (!"p_raw" %in% names(table)) return(table)
  if (!method %in% stats::p.adjust.methods) {
    stop("Unsupported P-value adjustment method: ", method, call. = FALSE)
  }
  if (is.null(by) || length(by) == 0L) {
    table$p_adjusted <- stats::p.adjust(table$p_raw, method = method)
    return(table)
  }
  missing <- setdiff(by, names(table))
  if (length(missing) > 0L) {
    stop(
      "Numerical-effect adjustment columns are absent: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
  grouping <- interaction(table[by], drop = TRUE, lex.order = TRUE)
  adjusted <- rep(NA_real_, nrow(table))
  for (group in levels(grouping)) {
    index <- which(grouping == group)
    adjusted[index] <- stats::p.adjust(table$p_raw[index], method = method)
  }
  table$p_adjusted <- adjusted
  table
}

.lfp_stats_validate_numerical_inputs <- function(model, data, by, test_var) {
  .lfp_stats_validate_name(test_var, "test_var")
  if (!test_var %in% names(data)) {
    stop("Numerical predictor is absent from data: ", test_var, call. = FALSE)
  }
  if (!is.numeric(data[[test_var]])) {
    stop("Numerical predictor must be numeric: ", test_var, call. = FALSE)
  }
  finite_unique <- unique(data[[test_var]][is.finite(data[[test_var]])])
  if (length(finite_unique) < 2L) {
    stop("Numerical predictor must have at least two finite values.", call. = FALSE)
  }
  model_vars <- all.vars(stats::formula(model))
  if (!test_var %in% model_vars) {
    stop("Numerical predictor is absent from the fitted model: ", test_var, call. = FALSE)
  }
  if (!is.null(by) && length(by) > 0L) {
    missing_data <- setdiff(by, names(data))
    missing_model <- setdiff(by, model_vars)
    if (length(missing_data) > 0L) {
      stop(
        "Numerical-effect strata are absent from data: ",
        paste(missing_data, collapse = ", "),
        call. = FALSE
      )
    }
    if (length(missing_model) > 0L) {
      stop(
        "Numerical-effect strata are absent from the fitted model: ",
        paste(missing_model, collapse = ", "),
        call. = FALSE
      )
    }
  }
  invisible(TRUE)
}

run_numerical_effects <- function(
    model,
    data,
    by = NULL,
    test_var,
    slope_at = 0,
    grid_probs = seq(0, 1, by = 0.1),
    grid_values = NULL,
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    p_adjust = "BH",
    p_adjust_by = NULL
) {
  .lfp_stats_check_packages("emmeans")
  weights <- match.arg(weights)
  .lfp_stats_validate_numerical_inputs(model, data, by, test_var)
  if (!is.numeric(slope_at) || length(slope_at) != 1L || !is.finite(slope_at)) {
    stop("`slope_at` must be one finite numeric value.", call. = FALSE)
  }

  by_is_null <- is.null(by) || length(by) == 0L
  by_formula <- if (by_is_null) {
    stats::as.formula("~ 1")
  } else {
    stats::as.formula(paste("~", paste(by, collapse = " * ")))
  }

  joint <- if (by_is_null) {
    emmeans::joint_tests(model)
  } else {
    emmeans::joint_tests(model, by = by)
  }
  omnibus <- as.data.frame(joint)
  if ("model term" %in% names(omnibus)) {
    names(omnibus)[names(omnibus) == "model term"] <- "term"
  }
  omnibus <- omnibus[omnibus$term == test_var, , drop = FALSE]
  if (nrow(omnibus) == 0L) {
    stop("No estimable joint test was returned for: ", test_var, call. = FALSE)
  }
  omnibus <- .lfp_stats_standardize_test_table(omnibus, p_name = "p_raw")
  omnibus <- .lfp_stats_adjust_numerical_table(
    omnibus,
    method = p_adjust,
    by = p_adjust_by
  )

  slope_grid <- emmeans::emtrends(
    model,
    specs = by_formula,
    var = test_var,
    at = stats::setNames(list(slope_at), test_var)
  )
  slope <- summary(
    slope_grid,
    infer = c(TRUE, TRUE),
    level = conf_level
  )
  slope <- .lfp_stats_standardize_test_table(slope, p_name = "p_raw")
  trend_col <- paste0(test_var, ".trend")
  if (!trend_col %in% names(slope) && "trend" %in% names(slope)) {
    trend_col <- "trend"
  }
  if (!trend_col %in% names(slope)) {
    stop("The local slope column is absent for: ", test_var, call. = FALSE)
  }
  slope$slope <- slope[[trend_col]]
  slope$eval_at <- slope_at
  slope <- .lfp_stats_adjust_numerical_table(
    slope,
    method = p_adjust,
    by = p_adjust_by
  )

  if (is.null(grid_values)) {
    if (!is.numeric(grid_probs) || length(grid_probs) < 2L ||
        any(!is.finite(grid_probs)) || any(grid_probs < 0 | grid_probs > 1)) {
      stop("`grid_probs` must contain at least two finite probabilities in [0, 1].", call. = FALSE)
    }
    grid_points <- as.numeric(
      stats::quantile(data[[test_var]], probs = grid_probs, na.rm = TRUE)
    )
  } else {
    if (!is.numeric(grid_values) || length(grid_values) < 2L ||
        any(!is.finite(grid_values))) {
      stop("`grid_values` must contain at least two finite numeric values.", call. = FALSE)
    }
    grid_points <- as.numeric(grid_values)
  }
  grid_points <- sort(unique(grid_points))
  if (length(grid_points) < 2L) {
    stop("The numerical EMM grid must contain at least two unique values.", call. = FALSE)
  }

  curve_formula <- if (by_is_null) {
    stats::as.formula(paste("~", test_var))
  } else {
    stats::as.formula(paste("~", test_var, "|", paste(by, collapse = " * ")))
  }
  curve_grid <- emmeans::emmeans(
    model,
    specs = curve_formula,
    at = stats::setNames(list(grid_points), test_var),
    weights = weights,
    level = conf_level
  )
  curve <- summary(curve_grid, infer = c(TRUE, FALSE), level = conf_level)
  curve <- .lfp_stats_normalize_emmean(
    .lfp_stats_normalize_ci(as.data.frame(curve))
  )

  rownames(omnibus) <- NULL
  rownames(slope) <- NULL
  rownames(curve) <- NULL
  list(
    omnibus = omnibus,
    slope = slope,
    curve = curve,
    grid_points = grid_points
  )
}
