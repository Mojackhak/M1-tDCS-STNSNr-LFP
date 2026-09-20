# Estimated marginal mean helpers for categorical mixed-model effects.

.lfp_stats_check_packages <- function(packages) {
  missing <- packages[
    !vapply(packages, requireNamespace, logical(1), quietly = TRUE)
  ]
  if (length(missing) > 0L) {
    stop(
      "Missing required R packages: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }
}

.lfp_stats_validate_name <- function(value, name) {
  if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
    stop("`", name, "` must be one non-empty string.", call. = FALSE)
  }
}

.lfp_stats_model_frame <- function(model) {
  tryCatch(
    stats::model.frame(model),
    error = function(error) {
      stop(
        "The fitted model does not expose a model frame: ", conditionMessage(error),
        call. = FALSE
      )
    }
  )
}

.lfp_stats_validate_axes <- function(model, x_var, panel_var = NULL, facet_var = NULL) {
  .lfp_stats_validate_name(x_var, "x_var")
  if (!is.null(panel_var)) .lfp_stats_validate_name(panel_var, "panel_var")
  if (!is.null(facet_var)) .lfp_stats_validate_name(facet_var, "facet_var")
  if (!is.null(facet_var) && is.null(panel_var)) {
    stop("`facet_var` requires `panel_var`.", call. = FALSE)
  }

  axes <- c(x_var, panel_var, facet_var)
  if (anyDuplicated(axes)) {
    stop("x, panel, and facet axes must be distinct.", call. = FALSE)
  }

  model_vars <- all.vars(stats::formula(model))
  missing <- setdiff(axes, model_vars)
  if (length(missing) > 0L) {
    stop(
      "Axes are absent from the fitted model: ", paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  model_frame <- .lfp_stats_model_frame(model)
  absent_frame <- setdiff(axes, names(model_frame))
  if (length(absent_frame) > 0L) {
    stop(
      "Axes are absent from the fitted model frame: ",
      paste(absent_frame, collapse = ", "),
      call. = FALSE
    )
  }
  nonfactor <- axes[!vapply(model_frame[axes], is.factor, logical(1))]
  if (length(nonfactor) > 0L) {
    stop(
      "Categorical EMM axes must be factors in the fitted model: ",
      paste(nonfactor, collapse = ", "),
      call. = FALSE
    )
  }
  invisible(model_frame)
}

.lfp_stats_emm_formula <- function(x_var, panel_var = NULL, facet_var = NULL) {
  if (is.null(panel_var)) {
    return(stats::reformulate(x_var))
  }
  rhs <- if (is.null(facet_var)) panel_var else paste(panel_var, facet_var, sep = " * ")
  stats::as.formula(paste("~", x_var, "|", rhs))
}

.lfp_stats_normalize_ci <- function(table) {
  if ("asymp.LCL" %in% names(table)) {
    names(table)[names(table) == "asymp.LCL"] <- "lower.CL"
  }
  if ("asymp.UCL" %in% names(table)) {
    names(table)[names(table) == "asymp.UCL"] <- "upper.CL"
  }
  if ("lower" %in% names(table) && !"lower.CL" %in% names(table)) {
    names(table)[names(table) == "lower"] <- "lower.CL"
  }
  if ("upper" %in% names(table) && !"upper.CL" %in% names(table)) {
    names(table)[names(table) == "upper"] <- "upper.CL"
  }
  table
}

.lfp_stats_normalize_emmean <- function(table) {
  if ("response" %in% names(table) && !"emmean" %in% names(table)) {
    names(table)[names(table) == "response"] <- "emmean"
  }
  table
}

.lfp_stats_add_statistic <- function(table) {
  if ("t.ratio" %in% names(table)) {
    table$statistic <- table$t.ratio
    table$statistic_type <- "t"
  } else if ("z.ratio" %in% names(table)) {
    table$statistic <- table$z.ratio
    table$statistic_type <- "z"
  } else if ("Chisq" %in% names(table)) {
    table$statistic <- table$Chisq
    table$statistic_type <- "Chisq"
  } else if ("F.ratio" %in% names(table)) {
    table$statistic <- table$F.ratio
    table$statistic_type <- "F"
  } else {
    table$statistic <- NA_real_
    table$statistic_type <- NA_character_
  }
  table
}

.lfp_stats_add_p_adjustment <- function(
    table,
    p_col,
    method = "none",
    by = NULL,
    output_col = "p_across"
) {
  if (!p_col %in% names(table)) {
    stop("P-value column is absent: ", p_col, call. = FALSE)
  }
  if (!method %in% stats::p.adjust.methods) {
    stop("Unsupported P-value adjustment method: ", method, call. = FALSE)
  }
  if (identical(method, "none")) {
    table[[output_col]] <- NA_real_
    return(table)
  }

  if (is.null(by) || length(by) == 0L) {
    table[[output_col]] <- stats::p.adjust(table[[p_col]], method = method)
    return(table)
  }

  missing <- setdiff(by, names(table))
  if (length(missing) > 0L) {
    stop(
      "P-value adjustment grouping columns are absent: ",
      paste(missing, collapse = ", "),
      call. = FALSE
    )
  }

  grouping <- interaction(table[by], drop = TRUE, lex.order = TRUE)
  adjusted <- rep(NA_real_, nrow(table))
  for (group in levels(grouping)) {
    index <- which(grouping == group)
    adjusted[index] <- stats::p.adjust(table[[p_col]][index], method = method)
  }
  table[[output_col]] <- adjusted
  table
}

.lfp_stats_standardize_test_table <- function(table, p_name = "p_raw") {
  table <- as.data.frame(table)
  table <- .lfp_stats_normalize_ci(table)
  table <- .lfp_stats_add_statistic(table)
  if ("p.value" %in% names(table)) {
    names(table)[names(table) == "p.value"] <- p_name
  }
  table
}

.lfp_stats_axis_levels <- function(emm_table, x_var) {
  axis <- emm_table[[x_var]]
  levels <- if (is.factor(axis)) levels(axis) else unique(as.character(axis))
  levels <- levels[levels %in% as.character(axis)]
  if (length(levels) < 2L) {
    stop("`x_var` must have at least two estimable levels.", call. = FALSE)
  }
  levels
}

.lfp_stats_attach_pair_groups <- function(table, x_levels, by) {
  pairs <- utils::combn(x_levels, 2L)
  pair_count <- ncol(pairs)

  if (is.null(by) || length(by) == 0L) {
    if (nrow(table) != pair_count) {
      stop("Unexpected number of pairwise contrasts.", call. = FALSE)
    }
    pair_index <- seq_len(pair_count)
  } else {
    grouping <- interaction(table[by], drop = TRUE, lex.order = TRUE)
    group_sizes <- table(grouping)
    if (any(group_sizes != pair_count)) {
      stop("Unexpected number of pairwise contrasts in an EMM stratum.", call. = FALSE)
    }
    pair_index <- ave(seq_len(nrow(table)), grouping, FUN = seq_along)
  }

  table$group1 <- pairs[1L, pair_index]
  table$group2 <- pairs[2L, pair_index]
  table$contrast <- paste(table$group1, "-", table$group2)
  table
}

.lfp_stats_apply_contrast_direction <- function(table, direction) {
  if (identical(direction, "reference_order")) return(table)

  old_group1 <- table$group1
  table$group1 <- table$group2
  table$group2 <- old_group1
  table$contrast <- paste(table$group1, "-", table$group2)

  for (column in intersect(c("estimate", "statistic", "t.ratio", "z.ratio"), names(table))) {
    table[[column]] <- -table[[column]]
  }
  if (all(c("lower.CL", "upper.CL") %in% names(table))) {
    old_lower <- table$lower.CL
    old_upper <- table$upper.CL
    table$lower.CL <- -old_upper
    table$upper.CL <- -old_lower
  }
  table
}

run_joint_term <- function(model, term = NULL, by = NULL) {
  .lfp_stats_check_packages("emmeans")
  joint <- if (is.null(by) || length(by) == 0L) {
    emmeans::joint_tests(model)
  } else {
    emmeans::joint_tests(model, by = by)
  }
  table <- as.data.frame(joint)
  if ("model term" %in% names(table)) {
    names(table)[names(table) == "model term"] <- "term"
  }
  table <- .lfp_stats_standardize_test_table(table, p_name = "p_raw")

  if (!is.null(term)) {
    missing <- setdiff(term, table$term)
    if (length(missing) > 0L) {
      stop("Joint-test terms are absent: ", paste(missing, collapse = ", "), call. = FALSE)
    }
    table <- table[table$term %in% term, , drop = FALSE]
  }
  rownames(table) <- NULL
  table
}

run_emm_pairwise <- function(
    model,
    x_var,
    panel_var = NULL,
    facet_var = NULL,
    at = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    within_adjust = "tukey",
    across_method = "none",
    across_on = c("p_raw", "p_within"),
    across_by = NULL,
    primary_p = c("p_within", "p_across", "p_raw"),
    contrast_direction = c("reference_order", "later_minus_earlier")
) {
  .lfp_stats_check_packages("emmeans")
  weights <- match.arg(weights)
  across_on <- match.arg(across_on)
  primary_p <- match.arg(primary_p)
  contrast_direction <- match.arg(contrast_direction)
  .lfp_stats_validate_axes(model, x_var, panel_var, facet_var)

  specs <- .lfp_stats_emm_formula(x_var, panel_var, facet_var)
  emm_grid <- emmeans::emmeans(
    model,
    specs = specs,
    at = at,
    weights = weights,
    level = conf_level
  )
  emm_table <- summary(emm_grid, type = "response", level = conf_level)
  emm_table <- as.data.frame(emm_table)
  emm_table <- .lfp_stats_normalize_ci(emm_table)
  emm_table <- .lfp_stats_normalize_emmean(emm_table)

  pair_grid <- pairs(emm_grid, adjust = "none")
  raw_table <- summary(
    pair_grid,
    type = "response",
    infer = c(TRUE, TRUE),
    adjust = "none"
  )
  within_table <- summary(
    pair_grid,
    type = "response",
    infer = c(TRUE, TRUE),
    adjust = within_adjust
  )
  raw_table <- .lfp_stats_standardize_test_table(raw_table, p_name = "p_raw")
  within_table <- .lfp_stats_standardize_test_table(
    within_table,
    p_name = "p_within"
  )

  by <- c(panel_var, facet_var)
  key <- c(by, "contrast")
  if (nrow(raw_table) != nrow(within_table) ||
      !identical(raw_table[key], within_table[key])) {
    stop("Raw and within-adjusted pairwise contrast tables are not aligned.", call. = FALSE)
  }

  contrast_table <- within_table
  contrast_table$p_raw <- raw_table$p_raw
  if (identical(within_adjust, "tukey")) {
    contrast_table$p_tukey <- contrast_table$p_within
  }
  x_levels <- .lfp_stats_axis_levels(emm_table, x_var)
  contrast_table <- .lfp_stats_attach_pair_groups(contrast_table, x_levels, by)
  contrast_table <- .lfp_stats_add_p_adjustment(
    contrast_table,
    p_col = across_on,
    method = across_method,
    by = across_by,
    output_col = "p_across"
  )

  if (identical(primary_p, "p_across") && identical(across_method, "none")) {
    stop("`primary_p = p_across` requires an across-strata adjustment.", call. = FALSE)
  }
  contrast_table$p_primary <- contrast_table[[primary_p]]
  contrast_table <- .lfp_stats_apply_contrast_direction(
    contrast_table,
    contrast_direction
  )
  rownames(emm_table) <- NULL
  rownames(contrast_table) <- NULL
  list(emm = emm_table, contrast = contrast_table)
}

run_emmean_vs_null <- function(
    model,
    x_var,
    panel_var = NULL,
    facet_var = NULL,
    null = 0,
    at = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    adjust_method = c("none", "holm"),
    primary_p = c("p_raw", "p_holm")
) {
  .lfp_stats_check_packages("emmeans")
  weights <- match.arg(weights)
  adjust_method <- match.arg(adjust_method)
  primary_p <- match.arg(primary_p)
  if (!is.numeric(null) || length(null) != 1L || !is.finite(null)) {
    stop("`null` must be one finite numeric value.", call. = FALSE)
  }
  .lfp_stats_validate_axes(model, x_var, panel_var, facet_var)

  specs <- .lfp_stats_emm_formula(x_var, panel_var, facet_var)
  emm_grid <- emmeans::emmeans(
    model,
    specs = specs,
    at = at,
    weights = weights,
    level = conf_level
  )
  emm_table <- summary(emm_grid, type = "response", level = conf_level)
  emm_table <- .lfp_stats_normalize_emmean(
    .lfp_stats_normalize_ci(as.data.frame(emm_table))
  )

  raw_table <- summary(
    emm_grid,
    null = null,
    infer = c(TRUE, TRUE),
    adjust = "none"
  )
  raw_table <- .lfp_stats_standardize_test_table(raw_table, p_name = "p_raw")
  test_table <- .lfp_stats_add_p_adjustment(
    raw_table,
    p_col = "p_raw",
    method = adjust_method,
    by = NULL,
    output_col = "p_holm"
  )
  if (identical(primary_p, "p_holm") && identical(adjust_method, "none")) {
    stop("`primary_p = p_holm` requires Holm adjustment.", call. = FALSE)
  }
  test_table$p_primary <- test_table[[primary_p]]
  test_table$null <- null
  rownames(emm_table) <- NULL
  rownames(test_table) <- NULL
  list(emm = emm_table, test = test_table)
}

run_interaction_contrast <- function(
    model,
    x_var,
    moderator_var,
    at = list(),
    weights = c("equal", "proportional"),
    conf_level = 0.95,
    x_method = "pairwise",
    moderator_method = "pairwise",
    adjust = "none"
) {
  .lfp_stats_check_packages("emmeans")
  weights <- match.arg(weights)
  .lfp_stats_validate_axes(model, x_var, panel_var = moderator_var)

  specs <- stats::as.formula(paste("~", x_var, "*", moderator_var))
  emm_grid <- emmeans::emmeans(
    model,
    specs = specs,
    at = at,
    weights = weights,
    level = conf_level
  )
  contrast_grid <- emmeans::contrast(
    emm_grid,
    interaction = c(x_method, moderator_method),
    adjust = "none"
  )
  raw_table <- summary(
    contrast_grid,
    infer = c(TRUE, TRUE),
    adjust = "none"
  )
  adjusted_table <- summary(
    contrast_grid,
    infer = c(TRUE, TRUE),
    adjust = adjust
  )
  raw_table <- .lfp_stats_standardize_test_table(raw_table, p_name = "p_raw")
  adjusted_table <- .lfp_stats_standardize_test_table(
    adjusted_table,
    p_name = "p_adjusted"
  )
  if (nrow(raw_table) != nrow(adjusted_table)) {
    stop("Raw and adjusted interaction contrast tables are not aligned.", call. = FALSE)
  }
  adjusted_table$p_raw <- raw_table$p_raw
  adjusted_table$p_primary <- if (identical(adjust, "none")) {
    adjusted_table$p_raw
  } else {
    adjusted_table$p_adjusted
  }
  rownames(adjusted_table) <- NULL
  adjusted_table
}
