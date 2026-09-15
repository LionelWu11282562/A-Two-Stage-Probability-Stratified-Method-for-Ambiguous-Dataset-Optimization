############################################
# Post-calibration using external hypertension prevalence
# Region-only version
#
# Method 1:
#   Province-level prevalence threshold calibration
#
# Method 2:
#   Province-level logit intercept shift calibration
#
# Confusion matrices:
#   1. Raw probability threshold 0.5
#   2. Logit-calibrated probability threshold 0.5
#   3. Province prevalence threshold calibration
############################################

library(readxl)
library(dplyr)
library(writexl)
library(ggplot2)
library(tidyr)

############################################
# 1. 路径设置
############################################

# 原始验证集：需要包含 sample_id、地区变量、真实结局 Z4
valid_full_path <- "/Users/huanqiwu/Desktop/FTestF.xlsx"

# XGBoost 输出的验证集概率文件：需要包含 sample_id 和 xgb_prob_A
valid_prob_path <- "/Users/huanqiwu/Desktop/XGBoost_results/tables/ValidationSet_with_XGBoost_prob_A.xlsx"

out_dir <- "/Users/huanqiwu/Desktop/PostCalibration_region_results"
fig_dir <- file.path(out_dir, "figures")
table_dir <- file.path(out_dir, "tables")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

############################################
# 2. 读取数据
############################################

valid_full <- read_excel(valid_full_path)
valid_prob <- read_excel(valid_prob_path)

############################################
# 3. 设置变量名
############################################

id_var <- "sample_id"

# 地区变量
province_var <- "Z2"

# 真实结局变量
# 如果你的真实 A/B/C 标签不是 Z4，而是 Z3，请改成 "Z3"
outcome_var <- "Z4"

# XGBoost 预测概率
prob_var <- "xgb_prob_A"

needed_full_vars <- c(id_var, province_var, outcome_var)
missing_full_vars <- setdiff(needed_full_vars, names(valid_full))

if (length(missing_full_vars) > 0) {
  stop(
    paste(
      "原始验证集中缺少变量：",
      paste(missing_full_vars, collapse = ", ")
    )
  )
}

if (!id_var %in% names(valid_prob)) {
  stop("概率文件中找不到 sample_id")
}

if (!prob_var %in% names(valid_prob)) {
  stop("概率文件中找不到 xgb_prob_A")
}

############################################
# 4. 合并原始验证集和预测概率
############################################

calib_data <- valid_full %>%
  left_join(
    valid_prob %>%
      dplyr::select(
        all_of(id_var),
        all_of(prob_var)
      ),
    by = id_var
  )

if (any(is.na(calib_data[[prob_var]]))) {
  warning("有样本没有成功匹配到预测概率，请检查 sample_id 是否一致。")
}

############################################
# 5. 外部地区总体患病率表
############################################

prevalence_table <- data.frame(
  province = c(
    "Prov_Beijing",
    "Prov_Guangdong",
    "Prov_Henan",
    "Prov_Jiangxi",
    "Prov_Jilin",
    "Prov_Xinjiang",
    "Prov_Yunnan",
    "Prov_Zhejiang"
  ),
  man = c(
    0.378,
    0.298,
    0.258,
    0.196,
    0.268,
    0.188,
    0.299,
    0.254
  ),
  women = c(
    0.341,
    0.247,
    0.223,
    0.151,
    0.256,
    0.176,
    0.269,
    0.210
  ),
  total = c(
    0.359,
    0.273,
    0.241,
    0.173,
    0.262,
    0.182,
    0.284,
    0.232
  )
)

# 只使用地区总体患病率 total
prevalence_region <- prevalence_table %>%
  dplyr::select(
    province,
    target_prev = total
  )

############################################
# 6. 处理地区变量并匹配外部患病率
############################################

calib_data <- calib_data %>%
  mutate(
    province_calib = as.character(.data[[province_var]])
  )

cat("地区分布：\n")
print(table(calib_data$province_calib, useNA = "ifany"))

calib_data <- calib_data %>%
  left_join(
    prevalence_region,
    by = c("province_calib" = "province")
  )

if (any(is.na(calib_data$target_prev))) {
  warning("部分样本没有匹配到外部地区患病率。请检查地区名称。")
  print(
    calib_data %>%
      filter(is.na(target_prev)) %>%
      count(province_calib)
  )
}

############################################
# 7. 构造真实标签：A = 1, B/C = 0
############################################

calib_data <- calib_data %>%
  mutate(
    y_true = case_when(
      .data[[outcome_var]] == "A" ~ 1,
      .data[[outcome_var]] %in% c("B", "C") ~ 0,
      .data[[outcome_var]] == 1 ~ 1,
      .data[[outcome_var]] %in% c(0, 2, 3) ~ 0,
      TRUE ~ NA_real_
    )
  )

cat("真实标签分布：\n")
print(table(calib_data$y_true, useNA = "ifany"))

############################################
# 8. 方法一：地区内按目标患病率做阈值校准
############################################
# 每个地区内：
# 按预测概率从高到低排序
# 取前 target_prev 比例作为预测阳性

threshold_calibrated <- calib_data %>%
  filter(!is.na(.data[[prob_var]]), !is.na(target_prev)) %>%
  group_by(province_calib) %>%
  arrange(desc(.data[[prob_var]]), .by_group = TRUE) %>%
  mutate(
    group_n = n(),
    target_n_positive = round(first(target_prev) * group_n),
    rank_in_group = row_number(),
    postcalib_positive_threshold = ifelse(
      rank_in_group <= target_n_positive,
      1,
      0
    ),
    group_threshold = case_when(
      target_n_positive <= 0 ~ Inf,
      target_n_positive >= group_n ~ -Inf,
      TRUE ~ sort(.data[[prob_var]], decreasing = TRUE)[target_n_positive]
    )
  ) %>%
  ungroup()

############################################
# 9. 方法二：地区内 logit intercept shift 概率校准
############################################
# 目标：
# 每个地区内 mean(prob_A_logit_calibrated) = target_prev

safe_logit <- function(p) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  log(p / (1 - p))
}

safe_inv_logit <- function(x) {
  1 / (1 + exp(-x))
}

find_delta <- function(p, target_prev) {
  p <- pmin(pmax(p, 1e-6), 1 - 1e-6)
  logit_p <- safe_logit(p)
  
  f <- function(delta) {
    mean(safe_inv_logit(logit_p + delta), na.rm = TRUE) - target_prev
  }
  
  uniroot(
    f,
    interval = c(-20, 20)
  )$root
}

delta_table <- calib_data %>%
  filter(!is.na(.data[[prob_var]]), !is.na(target_prev)) %>%
  group_by(province_calib) %>%
  summarise(
    n = n(),
    target_prev = first(target_prev),
    mean_raw_prob = mean(.data[[prob_var]], na.rm = TRUE),
    delta = find_delta(.data[[prob_var]], first(target_prev)),
    .groups = "drop"
  )

logit_calibrated <- calib_data %>%
  left_join(
    delta_table %>%
      dplyr::select(province_calib, delta),
    by = "province_calib"
  ) %>%
  mutate(
    prob_A_logit_calibrated = ifelse(
      !is.na(.data[[prob_var]]) & !is.na(delta),
      safe_inv_logit(safe_logit(.data[[prob_var]]) + delta),
      NA_real_
    )
  )

############################################
# 10. 合并两种校准结果
############################################

postcalib_data <- logit_calibrated %>%
  left_join(
    threshold_calibrated %>%
      dplyr::select(
        all_of(id_var),
        group_n,
        target_n_positive,
        rank_in_group,
        group_threshold,
        postcalib_positive_threshold
      ),
    by = id_var
  )

############################################
# 11. 生成不同方法的预测分类
############################################

postcalib_data <- postcalib_data %>%
  mutate(
    pred_raw_0.5 = case_when(
      !is.na(.data[[prob_var]]) & .data[[prob_var]] >= 0.5 ~ 1,
      !is.na(.data[[prob_var]]) & .data[[prob_var]] < 0.5 ~ 0,
      TRUE ~ NA_real_
    ),
    
    pred_logit_calibrated_0.5 = case_when(
      !is.na(prob_A_logit_calibrated) & prob_A_logit_calibrated >= 0.5 ~ 1,
      !is.na(prob_A_logit_calibrated) & prob_A_logit_calibrated < 0.5 ~ 0,
      TRUE ~ NA_real_
    ),
    
    pred_region_prevalence_threshold = case_when(
      !is.na(postcalib_positive_threshold) ~ as.numeric(postcalib_positive_threshold),
      TRUE ~ NA_real_
    )
  )

############################################
# 12. 混淆矩阵函数
############################################

make_confusion_matrix <- function(data, truth_col, pred_col, method_name) {
  
  tmp <- data %>%
    filter(
      !is.na(.data[[truth_col]]),
      !is.na(.data[[pred_col]])
    ) %>%
    mutate(
      truth = factor(.data[[truth_col]], levels = c(0, 1)),
      pred = factor(.data[[pred_col]], levels = c(0, 1))
    )
  
  cm <- table(
    Truth = tmp$truth,
    Predicted = tmp$pred
  )
  
  cm_df <- as.data.frame(cm) %>%
    mutate(
      method = method_name
    ) %>%
    dplyr::select(
      method,
      Truth,
      Predicted,
      Freq
    )
  
  TN <- cm["0", "0"]
  FP <- cm["0", "1"]
  FN <- cm["1", "0"]
  TP <- cm["1", "1"]
  
  safe_div <- function(a, b) {
    ifelse(b == 0, NA_real_, a / b)
  }
  
  accuracy <- safe_div(TP + TN, sum(cm))
  sensitivity <- safe_div(TP, TP + FN)
  specificity <- safe_div(TN, TN + FP)
  precision <- safe_div(TP, TP + FP)
  npv <- safe_div(TN, TN + FN)
  f1 <- safe_div(2 * precision * sensitivity, precision + sensitivity)
  balanced_accuracy <- mean(c(sensitivity, specificity), na.rm = TRUE)
  
  metrics_df <- data.frame(
    method = method_name,
    n = sum(cm),
    TP = as.numeric(TP),
    TN = as.numeric(TN),
    FP = as.numeric(FP),
    FN = as.numeric(FN),
    accuracy = as.numeric(accuracy),
    sensitivity_recall = as.numeric(sensitivity),
    specificity = as.numeric(specificity),
    precision_ppv = as.numeric(precision),
    npv = as.numeric(npv),
    f1 = as.numeric(f1),
    balanced_accuracy = as.numeric(balanced_accuracy)
  )
  
  list(
    confusion_matrix = cm_df,
    metrics = metrics_df
  )
}

############################################
# 13. 计算三种方式的混淆矩阵
############################################

cm_raw <- make_confusion_matrix(
  data = postcalib_data,
  truth_col = "y_true",
  pred_col = "pred_raw_0.5",
  method_name = "raw_probability_threshold_0.5"
)

cm_logit <- make_confusion_matrix(
  data = postcalib_data,
  truth_col = "y_true",
  pred_col = "pred_logit_calibrated_0.5",
  method_name = "region_logit_calibrated_threshold_0.5"
)

cm_region_prev <- make_confusion_matrix(
  data = postcalib_data,
  truth_col = "y_true",
  pred_col = "pred_region_prevalence_threshold",
  method_name = "region_prevalence_threshold_calibration"
)

confusion_matrix_all <- bind_rows(
  cm_raw$confusion_matrix,
  cm_logit$confusion_matrix,
  cm_region_prev$confusion_matrix
)

confusion_metrics_all <- bind_rows(
  cm_raw$metrics,
  cm_logit$metrics,
  cm_region_prev$metrics
)

print("混淆矩阵：")
print(confusion_matrix_all)

print("混淆矩阵指标：")
print(confusion_metrics_all)

############################################
# 14. 地区校准检查
############################################

calibration_check <- postcalib_data %>%
  filter(!is.na(target_prev)) %>%
  group_by(province_calib) %>%
  summarise(
    n = n(),
    target_prev = first(target_prev),
    raw_mean_prob = mean(.data[[prob_var]], na.rm = TRUE),
    logit_calibrated_mean_prob = mean(prob_A_logit_calibrated, na.rm = TRUE),
    threshold_positive_rate = mean(pred_region_prevalence_threshold, na.rm = TRUE),
    group_threshold = first(group_threshold),
    .groups = "drop"
  )

print("地区校准检查：")
print(calibration_check)

############################################
# 15. 保存结果表格
############################################

write_xlsx(
  list(
    postcalibrated_data = postcalib_data,
    prevalence_table = prevalence_table,
    prevalence_region = prevalence_region,
    delta_table = delta_table,
    calibration_check_by_province = calibration_check,
    confusion_matrix_all = confusion_matrix_all,
    confusion_metrics_all = confusion_metrics_all
  ),
  file.path(table_dir, "PostCalibration_region_results.xlsx")
)

write_xlsx(
  postcalib_data,
  file.path(table_dir, "ValidationSet_with_region_postcalibrated_probability.xlsx")
)

############################################
# 16. 绘图：校准前后概率分布
############################################

p_raw <- ggplot(
  postcalib_data,
  aes(
    x = .data[[prob_var]]
  )
) +
  geom_density(alpha = 0.4, fill = "grey70") +
  facet_wrap(~ province_calib) +
  labs(
    title = "Raw XGBoost Probability by Province",
    x = "Raw predicted probability of A",
    y = "Density"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "raw_probability_by_province.png"),
  plot = p_raw,
  width = 14,
  height = 8,
  dpi = 300
)

p_calib <- ggplot(
  postcalib_data,
  aes(
    x = prob_A_logit_calibrated
  )
) +
  geom_density(alpha = 0.4, fill = "grey70") +
  facet_wrap(~ province_calib) +
  labs(
    title = "Region Logit-Calibrated Probability by Province",
    x = "Region logit-calibrated probability of A",
    y = "Density"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "region_logit_calibrated_probability_by_province.png"),
  plot = p_calib,
  width = 14,
  height = 8,
  dpi = 300
)

p_check <- calibration_check %>%
  tidyr::pivot_longer(
    cols = c(
      raw_mean_prob,
      logit_calibrated_mean_prob,
      threshold_positive_rate,
      target_prev
    ),
    names_to = "metric",
    values_to = "value"
  ) %>%
  ggplot(
    aes(
      x = province_calib,
      y = value,
      fill = metric
    )
  ) +
  geom_col(position = "dodge") +
  coord_flip() +
  labs(
    title = "Region-level Calibration Check",
    x = "Province",
    y = "Rate / Mean probability",
    fill = "Metric"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "region_calibration_check.png"),
  plot = p_check,
  width = 10,
  height = 8,
  dpi = 300
)

############################################
# 17. 绘制混淆矩阵热图
############################################

p_cm <- confusion_matrix_all %>%
  mutate(
    Truth = factor(
      Truth,
      levels = c(0, 1),
      labels = c("True BC", "True A")
    ),
    Predicted = factor(
      Predicted,
      levels = c(0, 1),
      labels = c("Predicted BC", "Predicted A")
    )
  ) %>%
  ggplot(
    aes(
      x = Predicted,
      y = Truth,
      fill = Freq
    )
  ) +
  geom_tile() +
  geom_text(
    aes(label = Freq),
    size = 5
  ) +
  facet_wrap(~ method) +
  labs(
    title = "Confusion Matrices for Region-level Post-calibration Methods",
    x = "Predicted Label",
    y = "True Label",
    fill = "Count"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "confusion_matrices_region_postcalibration_methods.png"),
  plot = p_cm,
  width = 12,
  height = 5,
  dpi = 300
)

############################################
# 18. 完成提示
############################################

cat("地区-only 后校准完成。\n")
cat("核心输出文件：", file.path(table_dir, "PostCalibration_region_results.xlsx"), "\n")
cat("带后校准概率的数据：", file.path(table_dir, "ValidationSet_with_region_postcalibrated_probability.xlsx"), "\n")
cat("混淆矩阵图：", file.path(fig_dir, "confusion_matrices_region_postcalibration_methods.png"), "\n")
cat("地区校准检查图：", file.path(fig_dir, "region_calibration_check.png"), "\n")
