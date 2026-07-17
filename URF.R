############################################
# Random Forest on U-only Dataset
# Main model target:
#   A = 1
#   B/C = 0
#
# Output:
#   p1a = OOB predicted probability of A
#
# Second-stage label based on p1a:
#   dN = below dU lower bound
#   dU = middle mixed region
#   dP = above dU upper bound
#
# Final output:
#   Merge p1a and dPU_label back to full training set
############################################

library(readxl)
library(dplyr)
library(ranger)
library(writexl)
library(ggplot2)
library(pROC)
library(PRROC)

############################################
# 1. Input / output paths
############################################

file_in_path <- "/Users/huanqiwu/Desktop/FirstLProb/PU_labeling_results/tables/TrainingSet_with_PU_label.xlsx"

out_dir <- "/Users/huanqiwu/Desktop/FirstLProb/RF_U_dataset_results"

fig_dir <- file.path(out_dir, "figures")
table_dir <- file.path(out_dir, "tables")
rdata_dir <- file.path(out_dir, "rdata")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(rdata_dir, recursive = TRUE, showWarnings = FALSE)

rf_u_result_path <- file.path(
  table_dir,
  "random_forest_U_dataset_probability_results.xlsx"
)

full_training_dpu_path <- file.path(
  table_dir,
  "TrainingSet_with_p1a_and_dPU_label.xlsx"
)

############################################
# 2. Read full training data with first-stage PU_label
############################################

data_pu <- readxl::read_excel(
  file_in_path,
  sheet = "data_with_PU_label"
)

if (!"PU_label" %in% names(data_pu)) {
  stop("数据中找不到 PU_label。请先运行第一阶段 N/U/P 打标签代码。")
}

if (!"sample_id" %in% names(data_pu)) {
  stop("数据中找不到 sample_id。无法把 U 样本结果合并回完整训练集。")
}

if (!"Z3" %in% names(data_pu)) {
  stop("数据中找不到 Z3。无法定义 A=1, B/C=0。")
}

############################################
# 3. Keep U-only dataset for Random Forest
############################################

u_data <- data_pu %>%
  dplyr::filter(PU_label == "U")

cat("U samples before cleaning:", nrow(u_data), "\n")
cat("Original Z3 distribution in U dataset:\n")
print(table(u_data$Z3))

############################################
# 4. Define RF outcome: A = 1, B/C = 0
############################################

u_data <- u_data %>%
  dplyr::mutate(
    y_rf = dplyr::case_when(
      Z3 == "A" ~ 1,
      Z3 %in% c("B", "C") ~ 0,
      TRUE ~ NA_real_
    ),
    y_rf_factor = factor(
      y_rf,
      levels = c(0, 1),
      labels = c("NonA", "A")
    )
  ) %>%
  dplyr::filter(!is.na(y_rf))

cat("RF outcome distribution in U dataset:\n")
print(table(u_data$y_rf_factor))

if (length(unique(u_data$y_rf_factor)) < 2) {
  stop("U dataset contains only one outcome class. Random Forest classification cannot be trained.")
}

############################################
# 5. Automatically define predictors
############################################
# Important:
# Do NOT include labels, outcome variables, probability variables, or leakage variables.
# If both raw X and X_z exist, keep X_z and remove raw X.
############################################

exclude_vars <- c(
  "sample_id",
  "Z3",
  "Y1",
  "Y2",
  "Y3",
  "PU_label",
  "dPU_label",
  "final_PU_label",
  "y_rf",
  "y_rf_factor",
  "prob_A",
  "prob_A_2P_minus_1",
  "p1a",
  "rf_u_probability"
)

candidate_vars <- setdiff(names(u_data), exclude_vars)

# If both X and X_z exist, remove raw X and keep X_z
z_vars <- grep("_z$", candidate_vars, value = TRUE)
raw_with_z <- sub("_z$", "", z_vars)

candidate_vars <- setdiff(candidate_vars, raw_with_z)

cat("Candidate predictor variables:", length(candidate_vars), "\n")

if ("Y1" %in% candidate_vars) {
  stop("错误：Y1 仍然被纳入自变量，请检查 exclude_vars。")
}

############################################
# 6. Build RF input data
############################################

rf_data <- u_data %>%
  dplyr::select(
    dplyr::all_of(c(
      "sample_id",
      "Z3",
      "PU_label",
      "y_rf",
      "y_rf_factor",
      candidate_vars
    ))
  ) %>%
  stats::na.omit()

cat("U samples after na.omit:", nrow(rf_data), "\n")
cat("Z3 distribution after cleaning:\n")
print(table(rf_data$Z3))
cat("RF outcome distribution after cleaning:\n")
print(table(rf_data$y_rf_factor))

if (length(unique(rf_data$y_rf_factor)) < 2) {
  stop("After removing missing values, U dataset contains only one outcome class.")
}

############################################
# 7. Remove non-predictive metadata variables
############################################

rf_model_input <- rf_data %>%
  dplyr::select(
    -sample_id,
    -Z3,
    -PU_label,
    -y_rf
  )

############################################
# 8. Remove predictors with <2 unique values
############################################

predictor_names <- setdiff(names(rf_model_input), "y_rf_factor")

var_level_check <- data.frame(
  variable = predictor_names,
  n_unique = sapply(
    rf_model_input[predictor_names],
    function(x) length(unique(stats::na.omit(x)))
  ),
  class = sapply(
    rf_model_input[predictor_names],
    function(x) paste(class(x), collapse = ";")
  ),
  row.names = NULL
)

removed_rf_vars <- var_level_check %>%
  dplyr::filter(n_unique < 2)

valid_rf_vars <- var_level_check %>%
  dplyr::filter(n_unique >= 2) %>%
  dplyr::pull(variable)

cat("Variables removed because they have <2 unique values:\n")
print(removed_rf_vars)

rf_model_input <- rf_model_input %>%
  dplyr::select(
    y_rf_factor,
    dplyr::all_of(valid_rf_vars)
  )

############################################
# 9. Build model matrix
############################################

x_all <- model.matrix(
  y_rf_factor ~ .,
  data = rf_model_input
)[, -1, drop = FALSE]

y_all <- rf_model_input$y_rf_factor

# Remove zero-variance dummy columns
col_sd <- apply(x_all, 2, sd)
x_all <- x_all[, col_sd > 0, drop = FALSE]

cat("Final RF model matrix dimension:\n")
print(dim(x_all))

if (ncol(x_all) < 1) {
  stop("模型矩阵中没有可用自变量，请检查变量筛选结果。")
}

rf_model_data <- data.frame(
  y = y_all,
  x_all,
  check.names = TRUE
)

############################################
# 10. Fit Random Forest
# Fixed mtry = p
############################################

set.seed(123)

mtry_value <- ncol(x_all)

if (ncol(x_all) < mtry_value) {
  stop(
    paste0(
      "最终进入 RF 的变量数只有 ", ncol(x_all),
      " 个，小于指定的 mtry = ", mtry_value,
      "。请降低 mtry 或检查变量筛选。"
    )
  )
}

rf_u_model <- ranger::ranger(
  y ~ .,
  data = rf_model_data,
  probability = TRUE,
  mtry = mtry_value,
  num.trees = 500,
  importance = "impurity",
  seed = 123
)

print(rf_u_model)

############################################
# 11. OOB probability for A
# This remains the main goal:
# p1a = probability leaning toward A
############################################

rf_data$p1a <- rf_u_model$predictions[, "A"]

############################################
# 12. Second-stage dN / dU / dP based on p1a
############################################
# RF target is still A = 1, B/C = 0.
# p1a still means probability leaning toward A.
#
# Updated rule:
# dN: p1a <= U-C group p1a median
# dU: U-C group p1a median < p1a < U-A group p1a 75th percentile
# dP: p1a >= U-A group p1a 75th percentile
############################################

dC_median <- rf_data %>%
  dplyr::filter(Z3 == "C") %>%
  dplyr::summarise(
    median_value = median(p1a, na.rm = TRUE)
  ) %>%
  dplyr::pull(median_value)

dA_q75 <- rf_data %>%
  dplyr::filter(Z3 == "A") %>%
  dplyr::summarise(
    q75 = quantile(p1a, probs = 0.75, na.rm = TRUE)
  ) %>%
  dplyr::pull(q75)

if (length(dC_median) == 0 || is.na(dC_median)) {
  stop("无法计算 U 样本中 C 组 p1a 的中位数。请检查 U 样本中是否存在 C 组。")
}

if (length(dA_q75) == 0 || is.na(dA_q75)) {
  stop("无法计算 U 样本中 A 组 p1a 的第 75 分位数。请检查 U 样本中是否存在 A 组。")
}

d_threshold_table <- data.frame(
  threshold_name = c(
    "U_C_group_p1a_median_dU_lower_bound",
    "U_A_group_p1a_75th_percentile_dU_upper_bound"
  ),
  threshold_value = c(
    dC_median,
    dA_q75
  )
)

print("Second-stage dU thresholds based on p1a:")
print(d_threshold_table)

if (dC_median >= dA_q75) {
  warning("注意：dC_median >= dA_q75，dU 区间异常，请检查 U 样本中的 p1a 分布。")
}

rf_data <- rf_data %>%
  dplyr::mutate(
    dPU_label = dplyr::case_when(
      p1a <= dC_median ~ "dN",
      p1a > dC_median & p1a < dA_q75 ~ "dU",
      p1a >= dA_q75 ~ "dP",
      TRUE ~ NA_character_
    ),
    dPU_label = factor(dPU_label, levels = c("dN", "dU", "dP"))
  )

############################################
# 13. dPU label summary
############################################

dpu_count <- rf_data %>%
  dplyr::count(dPU_label, name = "n") %>%
  dplyr::mutate(
    proportion = n / sum(n)
  )

dpu_count_by_Z3 <- rf_data %>%
  dplyr::count(Z3, dPU_label, name = "n") %>%
  dplyr::group_by(Z3) %>%
  dplyr::mutate(
    within_group_proportion = n / sum(n)
  ) %>%
  dplyr::ungroup()

print("dN / dU / dP overall distribution:")
print(dpu_count)

print("dN / dU / dP distribution by Z3:")
print(dpu_count_by_Z3)

############################################
# 14. Merge p1a and dPU_label back to full training set
############################################

dpu_by_id <- rf_data %>%
  dplyr::select(
    sample_id,
    p1a,
    dPU_label
  )

full_training_with_dPU <- data_pu %>%
  dplyr::left_join(
    dpu_by_id,
    by = "sample_id"
  ) %>%
  dplyr::mutate(
    final_PU_label = dplyr::case_when(
      PU_label == "N" ~ "N",
      PU_label == "P" ~ "P",
      PU_label == "U" ~ as.character(dPU_label),
      TRUE ~ as.character(PU_label)
    ),
    final_PU_label = factor(
      final_PU_label,
      levels = c("N", "dN", "dU", "dP", "P")
    )
  )

final_label_count <- full_training_with_dPU %>%
  dplyr::count(final_PU_label, name = "n") %>%
  dplyr::mutate(
    proportion = n / sum(n)
  )

final_label_count_by_Z3 <- full_training_with_dPU %>%
  dplyr::count(Z3, final_PU_label, name = "n") %>%
  dplyr::group_by(Z3) %>%
  dplyr::mutate(
    within_group_proportion = n / sum(n)
  ) %>%
  dplyr::ungroup()

print("Final label distribution:")
print(final_label_count)

print("Final label distribution by Z3:")
print(final_label_count_by_Z3)

############################################
# 15. Prepare U dataset with p1a and dPU_label
############################################

u_with_p1a_dPU <- rf_data %>%
  dplyr::select(
    sample_id,
    Z3,
    PU_label,
    y_rf,
    y_rf_factor,
    p1a,
    dPU_label,
    dplyr::all_of(candidate_vars)
  )

############################################
# 16. Probability summary by Z3
############################################

p1a_summary_by_Z3 <- rf_data %>%
  dplyr::group_by(Z3) %>%
  dplyr::summarise(
    n = dplyr::n(),
    mean_p1a = mean(p1a, na.rm = TRUE),
    sd_p1a = sd(p1a, na.rm = TRUE),
    median_p1a = median(p1a, na.rm = TRUE),
    q10 = quantile(p1a, 0.10, na.rm = TRUE),
    q25 = quantile(p1a, 0.25, na.rm = TRUE),
    q75 = quantile(p1a, 0.75, na.rm = TRUE),
    q90 = quantile(p1a, 0.90, na.rm = TRUE),
    min_p1a = min(p1a, na.rm = TRUE),
    max_p1a = max(p1a, na.rm = TRUE),
    .groups = "drop"
  )

############################################
# 17. ROC / PR using p1a
############################################

roc_obj <- pROC::roc(
  response = rf_data$y_rf,
  predictor = rf_data$p1a,
  quiet = TRUE
)

auc_value <- as.numeric(pROC::auc(roc_obj))

pr_obj <- PRROC::pr.curve(
  scores.class0 = rf_data$p1a[rf_data$y_rf == 1],
  scores.class1 = rf_data$p1a[rf_data$y_rf == 0],
  curve = TRUE
)

pr_auc_value <- pr_obj$auc.integral

############################################
# 18. Variable importance
############################################

rf_importance <- data.frame(
  variable = names(rf_u_model$variable.importance),
  importance = as.numeric(rf_u_model$variable.importance),
  row.names = NULL
) %>%
  dplyr::arrange(desc(importance))

############################################
# 19. Variable inclusion summary
############################################

variable_inclusion_summary <- data.frame(
  item = c(
    "candidate_predictors_before_unique_filter",
    "valid_predictors_after_unique_filter",
    "dummy_columns_after_model_matrix",
    "dummy_columns_after_zero_variance_filter",
    "mtry_used"
  ),
  value = c(
    length(candidate_vars),
    length(valid_rf_vars),
    length(col_sd),
    ncol(x_all),
    mtry_value
  )
)

############################################
# 20. Save main tables
############################################

writexl::write_xlsx(
  u_with_p1a_dPU,
  file.path(table_dir, "U_dataset_with_RF_OOB_p1a_dPU_label.xlsx")
)

writexl::write_xlsx(
  list(
    full_training_with_dPU = full_training_with_dPU,
    U_dataset_with_p1a_dPU = u_with_p1a_dPU,
    d_threshold_table = d_threshold_table,
    dPU_count = dpu_count,
    dPU_count_by_Z3 = dpu_count_by_Z3,
    final_label_count = final_label_count,
    final_label_count_by_Z3 = final_label_count_by_Z3
  ),
  full_training_dpu_path
)

writexl::write_xlsx(
  rf_data %>% dplyr::filter(dPU_label == "dN"),
  file.path(table_dir, "dN_only_dataset.xlsx")
)

writexl::write_xlsx(
  rf_data %>% dplyr::filter(dPU_label == "dU"),
  file.path(table_dir, "dU_only_dataset.xlsx")
)

writexl::write_xlsx(
  rf_data %>% dplyr::filter(dPU_label == "dP"),
  file.path(table_dir, "dP_only_dataset.xlsx")
)

############################################
# 21. Plots
############################################

# p1a distribution by original Z3
p_density_Z3 <- ggplot(
  rf_data,
  aes(x = p1a, fill = Z3)
) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = dC_median, linetype = "dashed") +
  geom_vline(xintercept = dA_q75, linetype = "dashed") +
  labs(
    title = "Random Forest Probability Distribution in U Dataset",
    subtitle = "p1a = probability leaning toward A; dashed lines = dU bounds",
    x = "OOB predicted probability of A",
    y = "Density",
    fill = "Z3"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "rf_U_probability_density_by_Z3_with_dU_bounds.png"),
  p_density_Z3,
  width = 8,
  height = 6,
  dpi = 300
)

# p1a distribution by second-stage label
p_density_dPU <- ggplot(
  rf_data,
  aes(x = p1a, fill = dPU_label)
) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = dC_median, linetype = "dashed") +
  geom_vline(xintercept = dA_q75, linetype = "dashed") +
  labs(
    title = "Second-stage Stratification Based on p1a",
    subtitle = "dN / dU / dP are assigned after RF probability estimation",
    x = "OOB predicted probability of A",
    y = "Density",
    fill = "dPU Label"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "p1a_distribution_by_dPU_label.png"),
  p_density_dPU,
  width = 8,
  height = 6,
  dpi = 300
)

# Violin + boxplot by Z3
p_box_Z3 <- ggplot(
  rf_data,
  aes(x = Z3, y = p1a, fill = Z3)
) +
  geom_violin(alpha = 0.4, trim = FALSE) +
  geom_boxplot(width = 0.15, outlier.alpha = 0.3) +
  geom_hline(yintercept = dC_median, linetype = "dashed") +
  geom_hline(yintercept = dA_q75, linetype = "dashed") +
  labs(
    title = "Random Forest p1a in U Dataset",
    subtitle = "p1a still represents probability leaning toward A",
    x = "Z3 Group",
    y = "OOB predicted probability of A"
  ) +
  theme_bw() +
  theme(legend.position = "none")

ggsave(
  file.path(fig_dir, "rf_U_probability_violin_boxplot_by_Z3_with_dU_bounds.png"),
  p_box_Z3,
  width = 7,
  height = 6,
  dpi = 300
)

# Ranked p1a plot
rf_rank_df <- rf_data %>%
  dplyr::arrange(p1a) %>%
  dplyr::mutate(rank = dplyr::row_number())

p_rank <- ggplot(
  rf_rank_df,
  aes(x = rank, y = p1a, color = Z3)
) +
  geom_point(alpha = 0.5, size = 1) +
  geom_hline(yintercept = dC_median, linetype = "dashed") +
  geom_hline(yintercept = dA_q75, linetype = "dashed") +
  labs(
    title = "Ranked p1a in U Dataset",
    subtitle = "Samples ranked by probability leaning toward A",
    x = "Samples ranked by p1a",
    y = "OOB predicted probability of A",
    color = "Z3"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "rf_U_ranked_probability_by_Z3_with_dU_bounds.png"),
  p_rank,
  width = 9,
  height = 6,
  dpi = 300
)

# ROC curve
roc_df <- data.frame(
  specificity = roc_obj$specificities,
  sensitivity = roc_obj$sensitivities
) %>%
  dplyr::mutate(
    fpr = 1 - specificity
  )

p_roc <- ggplot(
  roc_df,
  aes(x = fpr, y = sensitivity)
) +
  geom_line(linewidth = 1) +
  geom_abline(
    intercept = 0,
    slope = 1,
    linetype = "dashed"
  ) +
  labs(
    title = "OOB ROC Curve for Random Forest in U Dataset",
    subtitle = paste0("AUC = ", round(auc_value, 4)),
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "rf_U_oob_roc_curve.png"),
  p_roc,
  width = 7,
  height = 6,
  dpi = 300
)

# PR curve
pr_df <- data.frame(
  recall = pr_obj$curve[, 1],
  precision = pr_obj$curve[, 2],
  threshold = pr_obj$curve[, 3]
)

p_pr <- ggplot(
  pr_df,
  aes(x = recall, y = precision)
) +
  geom_line(linewidth = 1) +
  labs(
    title = "OOB PR Curve for Random Forest in U Dataset",
    subtitle = paste0("PR-AUC = ", round(pr_auc_value, 4)),
    x = "Recall",
    y = "Precision"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "rf_U_oob_pr_curve.png"),
  p_pr,
  width = 7,
  height = 6,
  dpi = 300
)

# Variable importance
p_importance <- rf_importance %>%
  dplyr::slice_head(n = 30) %>%
  ggplot(
    aes(
      x = reorder(variable, importance),
      y = importance
    )
  ) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 30 Random Forest Variable Importance in U Dataset",
    x = "Variable",
    y = "Importance"
  ) +
  theme_bw()

ggsave(
  file.path(fig_dir, "rf_U_top30_variable_importance.png"),
  p_importance,
  width = 8,
  height = 10,
  dpi = 300
)

############################################
# 22. Save final workbook
############################################

writexl::write_xlsx(
  list(
    U_dataset_with_p1a_dPU = u_with_p1a_dPU,
    full_training_with_dPU = full_training_with_dPU,
    p1a_summary_by_Z3 = p1a_summary_by_Z3,
    d_threshold_table = d_threshold_table,
    dPU_count = dpu_count,
    dPU_count_by_Z3 = dpu_count_by_Z3,
    final_label_count = final_label_count,
    final_label_count_by_Z3 = final_label_count_by_Z3,
    variable_inclusion_summary = variable_inclusion_summary,
    removed_low_variation_variables = removed_rf_vars,
    variable_level_check = var_level_check,
    variable_importance = rf_importance,
    model_performance_OOB = data.frame(
      auc = auc_value,
      pr_auc = pr_auc_value,
      n_U_samples_used = nrow(rf_data),
      n_A = sum(rf_data$y_rf == 1),
      n_BC = sum(rf_data$y_rf == 0),
      dC_median = dC_median,
      dA_q75 = dA_q75,
      mtry_used = mtry_value
    )
  ),
  rf_u_result_path
)

############################################
# 23. Save model and data
############################################

saveRDS(
  rf_u_model,
  file.path(rdata_dir, "random_forest_U_dataset_model.rds")
)

saveRDS(
  rf_data,
  file.path(rdata_dir, "random_forest_U_dataset_probability_data_with_dPU.rds")
)

saveRDS(
  full_training_with_dPU,
  file.path(rdata_dir, "full_training_with_p1a_and_dPU_label.rds")
)

############################################
# 24. Print output information
############################################

cat("Random Forest on U dataset finished.\n")
cat("Model target: A = 1, B/C = 0.\n")
cat("p1a means probability leaning toward A.\n")
cat("dPU_label is only a second-stage stratification based on p1a.\n")
cat("Results saved in:\n")
cat(out_dir, "\n")
cat("U samples used:", nrow(rf_data), "\n")
cat("A samples:", sum(rf_data$y_rf == 1), "\n")
cat("B/C samples:", sum(rf_data$y_rf == 0), "\n")
cat("OOB AUC:", round(auc_value, 4), "\n")
cat("OOB PR-AUC:", round(pr_auc_value, 4), "\n")
cat("dU lower bound, U-C p1a q10:", dC_median, "\n")
cat("dU upper bound, U-A p1a q75:", dA_q75, "\n")
cat("U dataset with p1a and dPU_label:", file.path(table_dir, "U_dataset_with_RF_OOB_p1a_dPU_label.xlsx"), "\n")
cat("Full training set with p1a and dPU_label:", full_training_dpu_path, "\n")
cat("Final workbook:", rf_u_result_path, "\n")
cat("Density plot by Z3:", file.path(fig_dir, "rf_U_probability_density_by_Z3_with_dU_bounds.png"), "\n")
cat("Density plot by dPU:", file.path(fig_dir, "p1a_distribution_by_dPU_label.png"), "\n")