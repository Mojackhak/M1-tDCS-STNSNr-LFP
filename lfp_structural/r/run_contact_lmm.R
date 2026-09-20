# Fit the registered contact-level structural-connectivity mixed models.

arguments <- commandArgs(trailingOnly = TRUE)
if (length(arguments) != 3L) {
  stop(
    "Usage: Rscript run_contact_lmm.R MODEL_ROWS_CSV SETTINGS_JSON OUTPUT_DIR",
    call. = FALSE
  )
}

data_path <- arguments[[1L]]
settings_path <- arguments[[2L]]
output_dir <- arguments[[3L]]

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

settings <- jsonlite::fromJSON(settings_path, simplifyVector = TRUE)
model_rows <- readr::read_csv(data_path, show_col_types = FALSE, progress = FALSE)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
emmeans::emm_options(lmer.df = "kenward-roger")

phase_levels <- as.character(settings$phase_levels)
prediction_quantiles <- as.numeric(settings$prediction_quantiles)
model_formula <- stats::as.formula(settings$formula)
predictor_column <- if (is.null(settings$predictor_column)) {
  "X_z"
} else {
  as.character(settings$predictor_column)
}
slope_p_adjust <- if (is.null(settings$slope_p_adjust)) {
  "none"
} else {
  as.character(settings$slope_p_adjust)
}
slope_adjusted_p_column <- if (tolower(slope_p_adjust) == "holm") {
  "P_holm"
} else {
  "Q"
}
include_phase_slope_contrasts <- if (is.null(settings$phase_slope_contrasts)) {
  TRUE
} else {
  isTRUE(settings$phase_slope_contrasts)
}
if (length(predictor_column) != 1L || !predictor_column %in% names(model_rows)) {
  stop("The registered predictor column is absent from model rows.", call. = FALSE)
}
interaction_term <- paste0("Phase:", predictor_column)

model_fit_rows <- list()
fixed_effect_rows <- list()
omnibus_rows <- list()
phase_slope_rows <- list()
phase_contrast_rows <- list()
prediction_rows <- list()

append_row <- function(rows, value) {
  rows[[length(rows) + 1L]] <- value
  rows
}

empty_table <- function(columns) {
  as.data.frame(
    setNames(replicate(length(columns), character(), simplify = FALSE), columns),
    stringsAsFactors = FALSE
  )
}

bind_or_empty <- function(rows, columns) {
  if (length(rows) == 0L) {
    return(empty_table(columns))
  }
  dplyr::bind_rows(rows)
}

normalize_term <- function(value) {
  value <- as.character(value)
  value[value == paste0(predictor_column, ":Phase")] <- interaction_term
  value
}

fit_one <- function(data) {
  warnings <- character()
  model <- withCallingHandlers(
    lmerTest::lmer(
      formula = model_formula,
      data = data,
      REML = isTRUE(settings$reml)
    ),
    warning = function(warning) {
      warnings <<- c(warnings, conditionMessage(warning))
      invokeRestart("muffleWarning")
    }
  )
  optimizer_messages <- unlist(
    model@optinfo$conv$lme4$messages,
    recursive = TRUE,
    use.names = FALSE
  )
  optimizer_messages <- optimizer_messages[nzchar(optimizer_messages)]
  messages <- unique(c(warnings, optimizer_messages))
  convergence_pattern <- paste(
    "converg", "gradient", "hessian", "unable to evaluate",
    "unidentifiable", "eigenvalue", "rescale", sep = "|"
  )
  convergence <- messages[
    grepl(convergence_pattern, messages, ignore.case = TRUE)
  ]
  singular <- lme4::isSingular(model)
  status <- if (length(convergence) > 0L) {
    "convergence_warning"
  } else if (singular) {
    "singular"
  } else {
    "ok"
  }
  list(
    model = model,
    status = status,
    singular = singular,
    message = paste(messages, collapse = "; ")
  )
}

fixed_effect_table <- function(model, model_id) {
  coefficients <- as.data.frame(summary(model)$coefficients)
  coefficients$Term <- rownames(coefficients)
  rownames(coefficients) <- NULL
  names(coefficients)[names(coefficients) == "Estimate"] <- "Estimate"
  names(coefficients)[names(coefficients) == "Std. Error"] <- "SE"
  names(coefficients)[names(coefficients) == "t value"] <- "T"
  names(coefficients)[names(coefficients) == "Pr(>|t|)"] <- "P"
  critical <- ifelse(
    is.finite(coefficients$df),
    stats::qt(0.975, df = coefficients$df),
    stats::qnorm(0.975)
  )
  coefficients$Lower <- coefficients$Estimate - critical * coefficients$SE
  coefficients$Upper <- coefficients$Estimate + critical * coefficients$SE
  coefficients$ModelID <- model_id
  coefficients[, c("ModelID", "Term", "Estimate", "SE", "df", "Lower", "Upper", "T", "P")]
}

omnibus_table <- function(model, model_id) {
  output <- as.data.frame(emmeans::joint_tests(model))
  names(output)[names(output) == "model term"] <- "Term"
  names(output)[names(output) == "df1"] <- "NumeratorDF"
  names(output)[names(output) == "df2"] <- "DenominatorDF"
  names(output)[names(output) == "F.ratio"] <- "F"
  names(output)[names(output) == "p.value"] <- "P"
  output$Term <- normalize_term(output$Term)
  output <- output[
    output$Term %in% c("Phase", predictor_column, interaction_term),
    ,
    drop = FALSE
  ]
  output$ModelID <- model_id
  output[, c("ModelID", "Term", "NumeratorDF", "DenominatorDF", "F", "P")]
}

phase_slope_table <- function(model, model_id) {
  eval_at <- 0
  grid <- emmeans::emtrends(
    model,
    specs = ~ Phase,
    var = predictor_column,
    at = stats::setNames(list(eval_at), predictor_column)
  )
  output <- as.data.frame(
    summary(grid, infer = c(TRUE, TRUE), level = 0.95, adjust = "none")
  )
  names(output)[names(output) == paste0(predictor_column, ".trend")] <- "Slope"
  names(output)[names(output) == "lower.CL"] <- "Lower"
  names(output)[names(output) == "upper.CL"] <- "Upper"
  names(output)[names(output) == "t.ratio"] <- "T"
  names(output)[names(output) == "p.value"] <- "P"
  output[[slope_adjusted_p_column]] <- stats::p.adjust(
    output$P,
    method = slope_p_adjust
  )
  output$EvalAt <- eval_at
  output$Phase <- as.character(output$Phase)
  output$ModelID <- model_id
  output[
    ,
    c(
      "ModelID", "Phase", "Slope", "SE", "df", "Lower", "Upper", "T", "P",
      slope_adjusted_p_column, "EvalAt"
    )
  ]
}

phase_contrast_table <- function(model, model_id) {
  grid <- emmeans::emtrends(model, specs = ~ Phase, var = predictor_column)
  output <- as.data.frame(
    summary(
      emmeans::contrast(grid, method = "pairwise", adjust = "none"),
      infer = c(TRUE, TRUE),
      level = 0.95,
      adjust = "none"
    )
  )
  names(output)[names(output) == "estimate"] <- "Estimate"
  names(output)[names(output) == "lower.CL"] <- "Lower"
  names(output)[names(output) == "upper.CL"] <- "Upper"
  names(output)[names(output) == "t.ratio"] <- "T"
  names(output)[names(output) == "p.value"] <- "P"
  output$Contrast <- as.character(output$contrast)
  output$ModelID <- model_id
  output[, c("ModelID", "Contrast", "Estimate", "SE", "df", "Lower", "Upper", "T", "P")]
}

prediction_table <- function(model, data, model_id) {
  grid_values <- sort(unique(as.numeric(stats::quantile(
    data[[predictor_column]],
    probs = prediction_quantiles,
    na.rm = TRUE,
    names = FALSE
  ))))
  prediction_formula <- stats::as.formula(
    paste("~ Phase |", predictor_column)
  )
  grid <- emmeans::emmeans(
    model,
    specs = prediction_formula,
    at = stats::setNames(list(grid_values), predictor_column),
    weights = "equal"
  )
  output <- as.data.frame(summary(grid, infer = c(TRUE, FALSE), level = 0.95))
  names(output)[names(output) == "emmean"] <- "PredictedValue"
  names(output)[names(output) == "lower.CL"] <- "Lower"
  names(output)[names(output) == "upper.CL"] <- "Upper"
  output$Phase <- as.character(output$Phase)
  output$ModelID <- model_id
  output[
    ,
    c(
      "ModelID", "Phase", predictor_column, "PredictedValue", "SE", "df",
      "Lower", "Upper"
    )
  ]
}

model_ids <- sort(unique(as.character(model_rows$ModelID)))
for (model_id in model_ids) {
  data <- model_rows[model_rows$ModelID == model_id, , drop = FALSE]
  data$Phase <- factor(data$Phase, levels = phase_levels)
  data$ID <- factor(data$ID)
  data$ContactUnitID <- factor(data$ContactUnitID)

  fit_error <- tryCatch(fit_one(data), error = function(error) error)
  if (inherits(fit_error, "error")) {
    model_fit_rows <- append_row(
      model_fit_rows,
      data.frame(
        ModelID = model_id,
        FitStatus = "fit_error",
        FitMessage = conditionMessage(fit_error),
        Singular = NA,
        InferenceStatus = "not_run",
        InferenceMessage = "",
        AIC = NA_real_,
        BIC = NA_real_,
        LogLik = NA_real_,
        Deviance = NA_real_,
        Sigma = NA_real_,
        IDVariance = NA_real_,
        ContactUnitVariance = NA_real_,
        ResidualVariance = NA_real_,
        stringsAsFactors = FALSE
      )
    )
    next
  }

  fit <- fit_error
  model <- fit$model
  variance <- as.data.frame(lme4::VarCorr(model))
  id_variance <- variance$vcov[variance$grp == "ID"]
  contact_variance <- variance$vcov[
    grepl("ID:ContactUnitID|ContactUnitID:ID", variance$grp)
  ]
  residual_variance <- variance$vcov[variance$grp == "Residual"]

  fixed_effect_rows <- append_row(
    fixed_effect_rows,
    fixed_effect_table(model, model_id)
  )

  inference_messages <- character()
  omnibus_result <- tryCatch(
    omnibus_table(model, model_id),
    error = function(error) error
  )
  if (inherits(omnibus_result, "error")) {
    inference_messages <- c(inference_messages, conditionMessage(omnibus_result))
  } else {
    omnibus_rows <- append_row(omnibus_rows, omnibus_result)
  }

  slope_result <- tryCatch(
    phase_slope_table(model, model_id),
    error = function(error) error
  )
  if (inherits(slope_result, "error")) {
    inference_messages <- c(inference_messages, conditionMessage(slope_result))
  } else {
    phase_slope_rows <- append_row(phase_slope_rows, slope_result)
  }

  if (include_phase_slope_contrasts) {
    contrast_result <- tryCatch(
      phase_contrast_table(model, model_id),
      error = function(error) error
    )
    if (inherits(contrast_result, "error")) {
      inference_messages <- c(inference_messages, conditionMessage(contrast_result))
    } else {
      phase_contrast_rows <- append_row(phase_contrast_rows, contrast_result)
    }
  }

  prediction_result <- tryCatch(
    prediction_table(model, data, model_id),
    error = function(error) error
  )
  if (inherits(prediction_result, "error")) {
    inference_messages <- c(inference_messages, conditionMessage(prediction_result))
  } else {
    prediction_rows <- append_row(prediction_rows, prediction_result)
  }

  model_fit_rows <- append_row(
    model_fit_rows,
    data.frame(
      ModelID = model_id,
      FitStatus = fit$status,
      FitMessage = fit$message,
      Singular = fit$singular,
      InferenceStatus = if (length(inference_messages) == 0L) "ok" else "partial_error",
      InferenceMessage = paste(unique(inference_messages), collapse = "; "),
      AIC = stats::AIC(model),
      BIC = stats::BIC(model),
      LogLik = as.numeric(stats::logLik(model)),
      Deviance = stats::deviance(model),
      Sigma = stats::sigma(model),
      IDVariance = if (length(id_variance) == 1L) id_variance else NA_real_,
      ContactUnitVariance = if (length(contact_variance) == 1L) contact_variance else NA_real_,
      ResidualVariance = if (length(residual_variance) == 1L) residual_variance else NA_real_,
      stringsAsFactors = FALSE
    )
  )
}

readr::write_csv(
  bind_or_empty(
    model_fit_rows,
    c(
      "ModelID", "FitStatus", "FitMessage", "Singular", "InferenceStatus",
      "InferenceMessage", "AIC", "BIC", "LogLik", "Deviance", "Sigma",
      "IDVariance", "ContactUnitVariance", "ResidualVariance"
    )
  ),
  file.path(output_dir, "model_fit.csv"),
  na = ""
)
readr::write_csv(
  bind_or_empty(
    fixed_effect_rows,
    c("ModelID", "Term", "Estimate", "SE", "df", "Lower", "Upper", "T", "P")
  ),
  file.path(output_dir, "fixed_effects.csv"),
  na = ""
)
readr::write_csv(
  bind_or_empty(
    omnibus_rows,
    c("ModelID", "Term", "NumeratorDF", "DenominatorDF", "F", "P")
  ),
  file.path(output_dir, "omnibus_tests.csv"),
  na = ""
)
readr::write_csv(
  bind_or_empty(
    phase_slope_rows,
    c(
      "ModelID", "Phase", "Slope", "SE", "df", "Lower", "Upper", "T", "P",
      slope_adjusted_p_column, "EvalAt"
    )
  ),
  file.path(output_dir, "phase_slopes.csv"),
  na = ""
)
readr::write_csv(
  bind_or_empty(
    phase_contrast_rows,
    c("ModelID", "Contrast", "Estimate", "SE", "df", "Lower", "Upper", "T", "P")
  ),
  file.path(output_dir, "phase_slope_contrasts.csv"),
  na = ""
)
readr::write_csv(
  bind_or_empty(
    prediction_rows,
    c(
      "ModelID", "Phase", predictor_column, "PredictedValue", "SE", "df",
      "Lower", "Upper"
    )
  ),
  file.path(output_dir, "prediction_grid.csv"),
  na = ""
)
