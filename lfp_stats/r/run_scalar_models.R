# Fit configured scalar mixed models and return combined temporary tables.

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3L) {
  stop(
    "Usage: Rscript run_scalar_models.R MODEL_DATA_CSV PLAN_JSON OUTPUT_DIR",
    call. = FALSE
  )
}

data_path <- arguments[[1L]]
plan_path <- arguments[[2L]]
output_dir <- arguments[[3L]]
repo_root <- normalizePath(getwd(), mustWork = TRUE)

required_packages <- c("dplyr", "emmeans", "jsonlite", "lme4", "lmerTest", "readr")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop(
    "Missing required R packages: ", paste(missing_packages, collapse = ", "),
    call. = FALSE
  )
}

source(file.path(repo_root, "lfp_stats", "r", "emm.R"))
emmeans::emm_options(lmer.df = "kenward-roger")

plan <- jsonlite::fromJSON(plan_path, simplifyVector = FALSE)
model_data <- readr::read_csv(data_path, show_col_types = FALSE, progress = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

factor_levels <- lapply(plan$factor_levels, unlist, use.names = FALSE)
factor_columns <- unlist(plan$factor_columns, use.names = FALSE)
model_formula <- stats::as.formula(plan$model$formula)

model_status_rows <- list()
test_status_rows <- list()
joint_results <- list()
emm_results <- list()
tukey_results <- list()
null_test_results <- list()

add_identity <- function(table, model_id, test) {
  table$ModelID <- model_id
  table$TestKind <- test$kind
  table$TestID <- test$id
  table
}

append_test_status <- function(model_id, test, status, message) {
  test_status_rows[[length(test_status_rows) + 1L]] <<- data.frame(
    ModelID = model_id,
    TestKind = test$kind,
    TestID = test$id,
    Status = status,
    Message = message,
    stringsAsFactors = FALSE
  )
}

is_singular_test_error <- function(condition) {
  message <- conditionMessage(condition)
  grepl(
    "Lapack routine dgesv: system is exactly singular",
    message,
    fixed = TRUE
  ) || grepl("system is computationally singular", message, fixed = TRUE)
}

not_estimable_reason <- function(data) {
  for (factor_name in names(factor_levels)) {
    missing_levels <- setdiff(
      factor_levels[[factor_name]],
      unique(as.character(data[[factor_name]]))
    )
    if (length(missing_levels) > 0L) {
      return(paste(
        paste("Missing", factor_name, "levels:"),
        paste(missing_levels, collapse = ", ")
      ))
    }
  }
  if (length(unique(data$ID)) < 2L) {
    return("Fewer than two ID levels.")
  }
  fixed_formula <- lme4::nobars(model_formula)
  design <- stats::model.matrix(fixed_formula, data = data)
  if (qr(design)$rank < ncol(design)) {
    return("Fixed-effect design matrix is rank deficient.")
  }
  ""
}

fit_model <- function(data) {
  fit_warnings <- character()
  model <- withCallingHandlers(
    lmerTest::lmer(
      formula = model_formula,
      data = data,
      REML = isTRUE(plan$model$reml)
    ),
    warning = function(warning) {
      fit_warnings <<- c(fit_warnings, conditionMessage(warning))
      invokeRestart("muffleWarning")
    }
  )

  singular <- lme4::isSingular(model)
  optimizer_messages <- unlist(
    model@optinfo$conv$lme4$messages,
    recursive = TRUE,
    use.names = FALSE
  )
  optimizer_messages <- optimizer_messages[nzchar(optimizer_messages)]
  convergence_pattern <- paste(
    "converg",
    "gradient",
    "hessian",
    "unable to evaluate",
    "unidentifiable",
    "eigenvalue",
    "rescale",
    sep = "|"
  )
  warning_is_boundary <- grepl(
    "boundary.*singular",
    fit_warnings,
    ignore.case = TRUE
  )
  warning_is_convergence <- grepl(
    convergence_pattern,
    fit_warnings,
    ignore.case = TRUE
  )
  optimizer_is_boundary <- grepl(
    "boundary.*singular",
    optimizer_messages,
    ignore.case = TRUE
  )
  optimizer_is_convergence <- grepl(
    convergence_pattern,
    optimizer_messages,
    ignore.case = TRUE
  )
  unexpected_warnings <- fit_warnings[!(warning_is_boundary | warning_is_convergence)]
  if (length(unexpected_warnings) > 0L) {
    stop(
      "Unexpected lmer warning: ", paste(unique(unexpected_warnings), collapse = "; "),
      call. = FALSE
    )
  }
  unexpected_optimizer_messages <- optimizer_messages[
    !(optimizer_is_boundary | optimizer_is_convergence)
  ]
  if (length(unexpected_optimizer_messages) > 0L) {
    stop(
      "Unexpected lmer optimizer message: ",
      paste(unique(unexpected_optimizer_messages), collapse = "; "),
      call. = FALSE
    )
  }

  convergence_messages <- unique(c(
    optimizer_messages[optimizer_is_convergence],
    fit_warnings[warning_is_convergence]
  ))
  message <- paste(unique(c(
    fit_warnings[warning_is_boundary],
    optimizer_messages[optimizer_is_boundary],
    convergence_messages
  )), collapse = "; ")
  status <- if (length(convergence_messages) > 0L) {
    "convergence_warning"
  } else if (singular) {
    "singular"
  } else {
    "ok"
  }
  list(model = model, singular = singular, status = status, message = message)
}

for (model_entry in plan$models) {
  model_id <- model_entry$model_id
  data <- model_data[model_data$ModelID == model_id, , drop = FALSE]
  for (factor_name in names(factor_levels)) {
    data[[factor_name]] <- factor(
      data[[factor_name]],
      levels = factor_levels[[factor_name]]
    )
  }
  for (factor_column in factor_columns) {
    data[[factor_column]] <- factor(data[[factor_column]])
  }

  reason <- not_estimable_reason(data)
  if (nzchar(reason)) {
    model_status_rows[[length(model_status_rows) + 1L]] <- data.frame(
      ModelID = model_id,
      Singular = FALSE,
      Status = "not_estimable",
      Message = reason,
      stringsAsFactors = FALSE
    )
    for (test in plan$tests) {
      append_test_status(model_id, test, "not_estimable", reason)
    }
    next
  }

  fit <- fit_model(data)
  model_status_rows[[length(model_status_rows) + 1L]] <- data.frame(
    ModelID = model_id,
    Singular = fit$singular,
    Status = fit$status,
    Message = fit$message,
    stringsAsFactors = FALSE
  )

  for (test in plan$tests) {
    test_error <- tryCatch({
      if (identical(test$kind, "joint_term")) {
        result <- run_joint_term(fit$model, term = test$term)
        joint_results[[length(joint_results) + 1L]] <- add_identity(
          result,
          model_id,
          test
        )
      } else if (identical(test$kind, "emm_pairwise")) {
        panel_var <- if (is.null(test$panel_var)) NULL else test$panel_var
        facet_var <- if (is.null(test$facet_var)) NULL else test$facet_var
        result <- run_emm_pairwise(
          fit$model,
          x_var = test$x_var,
          panel_var = panel_var,
          facet_var = facet_var,
          weights = test$weights,
          conf_level = test$confidence_level,
          within_adjust = test$within_adjust,
          across_method = test$across_method,
          primary_p = "p_within",
          contrast_direction = test$contrast_direction
        )
        emm_results[[length(emm_results) + 1L]] <- add_identity(
          result$emm,
          model_id,
          test
        )
        tukey_results[[length(tukey_results) + 1L]] <- add_identity(
          result$contrast,
          model_id,
          test
        )
      } else if (identical(test$kind, "emmean_vs_null")) {
        panel_var <- if (is.null(test$panel_var)) NULL else test$panel_var
        facet_var <- if (is.null(test$facet_var)) NULL else test$facet_var
        result <- run_emmean_vs_null(
          fit$model,
          x_var = test$x_var,
          panel_var = panel_var,
          facet_var = facet_var,
          null = test$null,
          weights = test$weights,
          conf_level = test$confidence_level,
          adjust_method = test$adjust_method,
          primary_p = test$primary_p
        )
        emm_results[[length(emm_results) + 1L]] <- add_identity(
          result$emm,
          model_id,
          test
        )
        null_test_results[[length(null_test_results) + 1L]] <- add_identity(
          result$test,
          model_id,
          test
        )
      } else {
        stop("Unsupported test kind: ", test$kind, call. = FALSE)
      }
      NULL
    }, error = function(error) error)
    if (inherits(test_error, "error")) {
      if (!is_singular_test_error(test_error)) {
        stop(test_error)
      }
      append_test_status(
        model_id,
        test,
        "not_estimable",
        conditionMessage(test_error)
      )
      next
    }
    append_test_status(model_id, test, fit$status, fit$message)
  }
}

bind_results <- function(rows, empty_columns) {
  if (length(rows) == 0L) {
    empty <- as.data.frame(
      setNames(replicate(length(empty_columns), character(), simplify = FALSE), empty_columns),
      stringsAsFactors = FALSE
    )
    return(empty)
  }
  dplyr::bind_rows(rows)
}

readr::write_csv(
  dplyr::bind_rows(model_status_rows),
  file.path(output_dir, "model_status.csv"),
  na = ""
)
readr::write_csv(
  dplyr::bind_rows(test_status_rows),
  file.path(output_dir, "test_status.csv"),
  na = ""
)
readr::write_csv(
  bind_results(joint_results, c("ModelID", "TestKind", "TestID")),
  file.path(output_dir, "joint.csv"),
  na = ""
)
readr::write_csv(
  bind_results(emm_results, c("ModelID", "TestKind", "TestID")),
  file.path(output_dir, "emm.csv"),
  na = ""
)
readr::write_csv(
  bind_results(tukey_results, c("ModelID", "TestKind", "TestID")),
  file.path(output_dir, "tukey.csv"),
  na = ""
)
readr::write_csv(
  bind_results(null_test_results, c("ModelID", "TestKind", "TestID")),
  file.path(output_dir, "null_test.csv"),
  na = ""
)
