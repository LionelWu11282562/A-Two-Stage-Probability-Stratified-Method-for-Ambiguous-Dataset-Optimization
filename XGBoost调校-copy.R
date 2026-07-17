############################################
# XGBoost: Train on Training Set
# Tune on Validation Set
# Outcome: A = 1, B/C = 0
############################################

library(readxl)
library(dplyr)
library(writexl)
library(ggplot2)
library(xgboost)
library(pROC)

############################################
# 1. 路径设置
############################################


train_path <- "/Users/huanqiwu/Desktop/FTrainF.xlsx"
valid_path <- "/Users/huanqiwu/Desktop/FTestF.xlsx"
out_dir<-"/Users/huanqiwu/Desktop/XGBoost_results"
fig_dir <- file.path(out_dir, "figures")
table_dir <- file.path(out_dir, "tables")
model_dir <- file.path(out_dir, "model")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(model_dir, recursive = TRUE, showWarnings = FALSE)

############################################
# 2. 读取训练集和验证集
############################################

train_raw <- read_excel(train_path)
valid_raw <- read_excel(valid_path)

############################################
# 3. 定义变量
############################################

id_vars <- c("sample_id")

numeric_z_vars <- c(
  "X2_z", "X8_z", "X28_z", "X59_z",
  "X6_z", "X9_z", "X16_z", "X17_z", "X18_z", "X19_z",
  "X20_z", "X21_z", "X22_z", "X23_z", "X25_z",
  "X29_z", "X33_z", "X34_z", "X35_z", "X36_z",
  "X37_z", "X38_z", "X41_z", "X42_z", "X43_z"
)

#numeric_z_vars <- paste0(numeric_raw_vars, "_z")

binary_vars <- c(
  "X1", "X10", "X11", "X12", "X13", "X14", "X15",
  "X24", "X26", "X27", "X32", "X39", "X40",
  "X45", "X46", "X47", "X48", "X49", "X50",
  "X51", "X52", "X53", "X54", "X55", "X56",
  "X57", "X58"
)

multiclass_vars <- c(
  "Z2", "X3", "X4", "X5", "X7", "X30", "X31", "X44"
)

outcome_var <- "Z4"

predictor_vars <- c(
  binary_vars,
  multiclass_vars,
  numeric_z_vars
)



############################################
# 4. 汇总所有自变量并检查是否存在
############################################

# 将三类变量合并
all_potential_predictors <- c(
  binary_vars,
  multiclass_vars,
  numeric_z_vars
)

# 检查这些变量是否真的在 train_raw 的列名里
available_predictor_vars <- all_potential_predictors[all_potential_predictors %in% names(train_raw)]

# 打印一下，看看有没有变量因为名字写错被漏掉了
missing_vars <- setdiff(all_potential_predictors, available_predictor_vars)
if (length(missing_vars) > 0) {
  cat("警告：以下变量在原始数据中未找到：\n")
  print(missing_vars)
}


cat("最终进入 XGBoost 的原始 predictor 数量：", length(available_predictor_vars), "\n")

############################################
# 5. 准备结局变量 (使用原始 Z4)
############################################
train_data <- train_raw %>%
  filter(!is.na(.data[[outcome_var]])) %>%
  mutate(y = as.numeric(as.character(.data[[outcome_var]])))

valid_data <- valid_raw %>%
  filter(!is.na(.data[[outcome_var]])) %>%
  mutate(y = as.numeric(as.character(.data[[outcome_var]])))

cat("训练集样本数：", nrow(train_data), "\n")
cat("训练集分布：\n")
print(table(train_data$y))

cat("验证集样本数：", nrow(valid_data), "\n")
cat("验证集分布：\n")
print(table(valid_data$y))

# 检查类别数量
if (length(unique(train_data$y)) < 2) {
  stop("训练集中类别不足，无法进行二分类训练。")
}

############################################
# 6. 合并训练集和验证集，保证 dummy 编码一致
############################################

train_model <- train_data %>%
  dplyr::select(
    dplyr::all_of(c(id_vars, outcome_var, "y", available_predictor_vars))
  ) %>%
  mutate(dataset = "train")

valid_model <- valid_data %>%
  dplyr::select(
    dplyr::all_of(c(id_vars, outcome_var, "y", available_predictor_vars))
  ) %>%
  mutate(dataset = "valid")

combined_data <- bind_rows(train_model, valid_model)

############################################
# 7. 字符型变量转 factor，缺失类别设为 Missing
############################################

combined_data <- combined_data %>%
  mutate(
    across(
      all_of(available_predictor_vars),
      ~ {
        if (is.character(.x) || is.factor(.x)) {
          x_chr <- as.character(.x)
          x_chr[is.na(x_chr)] <- "Missing"
          as.factor(x_chr)
        } else {
          .x
        }
      }
    )
  )

############################################
# 8. 构建模型矩阵
############################################

x_formula <- as.formula(
  paste("~", paste(available_predictor_vars, collapse = " + "), "- 1")
)

x_all <- model.matrix(
  x_formula,
  data = combined_data
)

train_idx <- combined_data$dataset == "train"
valid_idx <- combined_data$dataset == "valid"

x_train <- x_all[train_idx, , drop = FALSE]
x_valid <- x_all[valid_idx, , drop = FALSE]

y_train <- combined_data$y[train_idx]
y_valid <- combined_data$y[valid_idx]

train_id <- combined_data$sample_id[train_idx]
valid_id <- combined_data$sample_id[valid_idx]

train_group <- combined_data[[outcome_var]][train_idx]
valid_group <- combined_data[[outcome_var]][valid_idx]

cat("XGBoost 训练矩阵维度：\n")
print(dim(x_train))

cat("XGBoost 验证矩阵维度：\n")
print(dim(x_valid))

############################################
# 9. 构建 XGBoost DMatrix
############################################

dtrain <- xgb.DMatrix(
  data = x_train,
  label = y_train,
  missing = NA
)

dvalid <- xgb.DMatrix(
  data = x_valid,
  label = y_valid,
  missing = NA
)

############################################
# 10. 使用固定 XGBoost 参数
############################################

fixed_params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 3,
  eta = 0.06,
  min_child_weight = 5,
  subsample = 0.5,
  colsample_bytree = 0.6
)

############################################
# 11. 用固定参数训练模型
############################################

set.seed(123)

best_model <- xgb.train(
  params = fixed_params,
  data = dtrain,
  nrounds = 2000,
  evals = list(
    train = dtrain,
    valid = dvalid
  ),
  early_stopping_rounds = 50,
  maximize = TRUE,
  verbose = 1
)

############################################
# 12. 记录固定参数结果
############################################

valid_prob_tmp <- predict(
  best_model,
  dvalid
)

valid_auc_tmp <- as.numeric(
  pROC::auc(
    pROC::roc(
      response = y_valid,
      predictor = valid_prob_tmp,
      quiet = TRUE
    )
  )
)

valid_logloss_tmp <- -mean(
  y_valid * log(pmax(valid_prob_tmp, 1e-15)) +
    (1 - y_valid) * log(pmax(1 - valid_prob_tmp, 1e-15))
)

safe_best_iter <- if (is.null(best_model$best_iteration)) {
  NA_integer_
} else {
  best_model$best_iteration
}

safe_best_score <- if (is.null(best_model$best_score)) {
  NA_real_
} else {
  best_model$best_score
}

best_result <- data.frame(
  model_id = 1,
  max_depth = fixed_params$max_depth,
  eta = fixed_params$eta,
  min_child_weight = fixed_params$min_child_weight,
  subsample = fixed_params$subsample,
  colsample_bytree = fixed_params$colsample_bytree,
  best_iteration = safe_best_iter,
  best_score_from_xgb = safe_best_score,
  valid_auc = valid_auc_tmp,
  valid_logloss = valid_logloss_tmp
)

tuning_results <- best_result

print("固定参数模型结果：")
print(best_result)

############################################
# 13. 输出训练集和验证集预测概率
############################################


# 移除 iterationrange，让模型自动处理
train_prob <- predict(
  best_model,
  dtrain
)

valid_prob <- predict(
  best_model,
  dvalid
)

# 提示：如果你非常执着于手动指定范围，必须这样写：
# b_iter <- if (is.null(best_model$best_iteration) || is.na(best_model$best_iteration)) 
#           (best_model$niter - 1) else best_model$best_iteration
# train_prob <- predict(best_model, dtrain, iterationrange = c(0L, as.integer(b_iter + 1L)))

train_auc <- as.numeric(
  pROC::auc(
    pROC::roc(
      response = y_train,
      predictor = train_prob,
      quiet = TRUE
    )
  )
)

valid_auc <- as.numeric(
  pROC::auc(
    pROC::roc(
      response = y_valid,
      predictor = valid_prob,
      quiet = TRUE
    )
  )
)

train_pred_table <- data.frame(
  sample_id = train_id,
  original_group = train_group,
  y = y_train,
  xgb_prob_A = train_prob,
  dataset = "train"
)

valid_pred_table <- data.frame(
  sample_id = valid_id,
  original_group = valid_group,
  y = y_valid,
  xgb_prob_A = valid_prob,
  dataset = "valid"
)

prediction_table <- bind_rows(
  train_pred_table,
  valid_pred_table
)

performance_table <- data.frame(
  dataset = c("train", "valid"),
  auc = c(train_auc, valid_auc),
  n = c(length(y_train), length(y_valid)),
  n_A = c(sum(y_train == 1), sum(y_valid == 1)),
  n_BC = c(sum(y_train == 0), sum(y_valid == 0))
)

print("模型性能：")
print(performance_table)

############################################
# 14. 变量重要性
############################################

importance_table <- xgb.importance(
  model = best_model,
  feature_names = colnames(x_train)
)

############################################
# 15. 保存结果表格
############################################

write_xlsx(
  list(
    best_parameters = best_result,
    tuning_results = tuning_results,
    performance = performance_table,
    predictions = prediction_table,
    variable_importance = importance_table
  ),
  file.path(table_dir, "XGBoost_train_valid_results.xlsx")
)

write_xlsx(
  train_pred_table,
  file.path(table_dir, "TrainingSet_with_XGBoost_prob_A.xlsx")
)

write_xlsx(
  valid_pred_table,
  file.path(table_dir, "ValidationSet_with_XGBoost_prob_A.xlsx")
)

############################################
# 16. 保存模型
############################################

xgb.save(
  best_model,
  file.path(model_dir, "best_xgboost_model.json")
)

saveRDS(
  list(
    best_model = best_model,
    best_result = best_result,
    tuning_results = tuning_results,
    predictor_vars = available_predictor_vars,
    feature_names = colnames(x_train)
  ),
  file.path(model_dir, "best_xgboost_model_object.rds")
)

############################################
# 17. 绘制验证集概率分布图
############################################

p_valid_density <- ggplot(
  valid_pred_table,
  aes(
    x = xgb_prob_A,
    fill = as.factor(original_group),  # 强制转为因子
    group = original_group             # 显式指定分组
  )
) +
  geom_density(alpha = 0.4) +
  labs(
    title = "XGBoost Predicted Probability Distribution on Validation Set",
    subtitle = "Trained on Training Set; tuned on Validation Set",
    x = "Predicted probability of A",
    y = "Density",
    fill = "Original Group"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "ValidationSet_XGBoost_probability_distribution_by_group.png"),
  plot = p_valid_density,
  width = 8,
  height = 6,
  dpi = 300
)

############################################
# 18. 绘制训练集概率分布图
############################################

p_train_density <- ggplot(
    valid_pred_table,
    aes(
      x = xgb_prob_A,
      fill = as.factor(original_group),  # 强制转为因子
      group = original_group             # 显式指定分组
    )
  ) +
  geom_density(alpha = 0.4) +
  labs(
    title = "XGBoost Predicted Probability Distribution on Training Set",
    x = "Predicted probability of A",
    y = "Density",
    fill = "Original Group"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "TrainingSet_XGBoost_probability_distribution_by_group.png"),
  plot = p_train_density,
  width = 8,
  height = 6,
  dpi = 300
)

############################################
# 19. 绘制变量重要性图
############################################

p_importance <- importance_table %>%
  slice_head(n = 30) %>%
  ggplot(
    aes(
      x = reorder(Feature, Gain),
      y = Gain
    )
  ) +
  geom_col() +
  coord_flip() +
  labs(
    title = "Top 30 XGBoost Feature Importance",
    x = "Feature",
    y = "Gain"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "XGBoost_top30_feature_importance.png"),
  plot = p_importance,
  width = 8,
  height = 10,
  dpi = 300
)

############################################
# 20. 完成提示
############################################

cat("XGBoost 完成。\n")
cat("训练集 AUC：", train_auc, "\n")
cat("验证集 AUC：", valid_auc, "\n")
cat("最佳参数：\n")
print(best_result)
cat("结果表：", file.path(table_dir, "XGBoost_train_valid_results.xlsx"), "\n")
cat("训练集预测：", file.path(table_dir, "TrainingSet_with_XGBoost_prob_A.xlsx"), "\n")
cat("验证集预测：", file.path(table_dir, "ValidationSet_with_XGBoost_prob_A.xlsx"), "\n")
cat("模型文件：", file.path(model_dir, "best_xgboost_model.json"), "\n")




############################################
# 21. 混淆矩阵与分类报告
############################################

# 设定阈值（通常为 0.5）
threshold <- 0.5

# 1. 验证集分类预测
valid_pred_class <- ifelse(valid_prob > threshold, 1, 0)

# 2. 生成混淆矩阵
# 使用 factor 确保即使某个类别没预测出来，矩阵维度也是对的
cm <- table(
  Actual = factor(y_valid, levels = c(0, 1), labels = c("B/C", "A")),
  Predicted = factor(valid_pred_class, levels = c(0, 1), labels = c("B/C", "A"))
)

cat("\n--- 验证集混淆矩阵 ---\n")
print(cm)

# 3. 计算关键指标
tp <- cm["A", "A"]       # 真阳性
tn <- cm["B/C", "B/C"]   # 真阴性
fp <- cm["B/C", "A"]     # 假阳性
fn <- cm["A", "B/C"]     # 假阴性

accuracy <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)  # 精确率：预测为A中真正是A的比例
recall <- tp / (tp + fn)     # 召回率：实际为A中被预测出来的比例
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\n--- 分类性能指标 (Threshold =", threshold, ") ---\n")
cat("准确率 (Accuracy): ", round(accuracy, 4), "\n")
cat("精确率 (Precision):", round(precision, 4), "\n")
cat("召回率 (Recall):   ", round(recall, 4), "\n")
cat("F1 分数 (F1 Score):", round(f1_score, 4), "\n")

###########################################
# 22. 调整阈值并重新进行分类 
###########################################
# 设定阈值（通常为 0.5）
threshold <- 0.5

# 1. 验证集分类预测
valid_pred_class <- ifelse(valid_prob > threshold, 1, 0)

# 2. 生成混淆矩阵
# 使用 factor 确保即使某个类别没预测出来，矩阵维度也是对的
cm <- table(
  Actual = factor(y_valid, levels = c(0, 1), labels = c("B/C", "A")),
  Predicted = factor(valid_pred_class, levels = c(0, 1), labels = c("B/C", "A"))
)

cat("\n--- 验证集混淆矩阵 ---\n")
print(cm)

# 3. 计算关键指标
tp <- cm["A", "A"]       # 真阳性
tn <- cm["B/C", "B/C"]   # 真阴性
fp <- cm["B/C", "A"]     # 假阳性
fn <- cm["A", "B/C"]     # 假阴性

accuracy <- (tp + tn) / sum(cm)
precision <- tp / (tp + fp)  # 精确率：预测为A中真正是A的比例
recall <- tp / (tp + fn)     # 召回率：实际为A中被预测出来的比例
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\n--- 分类性能指标 (Threshold =", threshold, ") ---\n")
cat("准确率 (Accuracy): ", round(accuracy, 4), "\n")
cat("精确率 (Precision):", round(precision, 4), "\n")
cat("召回率 (Recall):   ", round(recall, 4), "\n")
cat("F1 分数 (F1 Score):", round(f1_score, 4), "\n")


############################################
# 23. SHAP 分析：使用 shapviz 替代 SHAPforXGBoost
############################################

# 1. 检查并加载必要包
if (!requireNamespace("shapviz", quietly = TRUE)) {
  install.packages("shapviz")
}

library(shapviz)
library(ggplot2)
library(writexl)

cat("\n--- 正在启动 SHAP 解释性分析 ---\n")

############################################
# 2. 准备数据
############################################

# 注意：这里必须使用训练 XGBoost 时的同一套特征矩阵
# 如果你前面模型使用的是 X_train，就写 X_train
X_matrix <- as.matrix(X_train)

############################################
# 3. 生成 shapviz 对象
############################################

# 对 xgboost 模型，shapviz 可以直接读取模型并计算 SHAP
sv <- shapviz(
  object = xgb_model,   # 如果你的最终模型叫 best_model，就改成 best_model
  X_pred = X_matrix,
  X = as.data.frame(X_matrix)
)

############################################
# 4. SHAP Summary Plot / Beeswarm
############################################

p_shap_summary <- sv_importance(
  sv,
  kind = "bee"
) +
  labs(
    title = "SHAP Summary Plot: Impact on Outcome Z4 = 1",
    subtitle = "Feature value from low to high"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "XGBoost_SHAP_01_summary_plot.png"),
  plot = p_shap_summary,
  width = 10,
  height = 8,
  dpi = 300
)

############################################
# 5. SHAP Importance Bar Plot
############################################

p_shap_importance <- sv_importance(
  sv,
  kind = "bar",
  max_display = 20
) +
  labs(
    title = "Top 20 Features by Mean Absolute SHAP Value"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "XGBoost_SHAP_02_importance_bar.png"),
  plot = p_shap_importance,
  width = 8,
  height = 6,
  dpi = 300
)

############################################
# 6. 提取前 4 个重要变量
############################################

shap_values_matrix <- get_shap_values(sv)

shap_score_df <- data.frame(
  Feature = colnames(shap_values_matrix),
  Mean_Abs_SHAP = colMeans(abs(shap_values_matrix))
) %>%
  arrange(desc(Mean_Abs_SHAP))

top4_vars <- shap_score_df$Feature[1:4]

cat("\nTop 4 SHAP features:\n")
print(top4_vars)

############################################
# 7. 绘制前 4 个变量的 Dependence Plots
############################################

p_dep_1 <- sv_dependence(sv, v = top4_vars[1]) + theme_bw()
p_dep_2 <- sv_dependence(sv, v = top4_vars[2]) + theme_bw()
p_dep_3 <- sv_dependence(sv, v = top4_vars[3]) + theme_bw()
p_dep_4 <- sv_dependence(sv, v = top4_vars[4]) + theme_bw()

# 如果还没有 patchwork，就安装
if (!requireNamespace("patchwork", quietly = TRUE)) {
  install.packages("patchwork")
}
library(patchwork)

p_shap_dep <- (p_dep_1 | p_dep_2) / (p_dep_3 | p_dep_4)

ggsave(
  filename = file.path(fig_dir, "XGBoost_SHAP_03_dependence_top4.png"),
  plot = p_shap_dep,
  width = 12,
  height = 10,
  dpi = 300
)

############################################
# 8. 导出 SHAP 重要性数值表
############################################

write_xlsx(
  shap_score_df,
  file.path(table_dir, "XGBoost_SHAP_importance_values.xlsx")
)

cat("已生成图表：Summary Plot, Importance Bar, Dependence Plots\n")
cat("数据表已存至：", file.path(table_dir, "XGBoost_SHAP_importance_values.xlsx"), "\n")

############################################
# 补充：输出完整混淆矩阵指标
############################################

TP <- sum(y_valid == 1 & valid_pred_class == 1)
TN <- sum(y_valid == 0 & valid_pred_class == 0)
FP <- sum(y_valid == 0 & valid_pred_class == 1)
FN <- sum(y_valid == 1 & valid_pred_class == 0)

accuracy <- (TP + TN) / (TP + TN + FP + FN)
sensitivity_recall <- TP / (TP + FN)
specificity <- TN / (TN + FP)
precision_ppv <- TP / (TP + FP)
npv <- TN / (TN + FN)
f1 <- 2 * precision_ppv * sensitivity_recall /
  (precision_ppv + sensitivity_recall)
balanced_accuracy <- (sensitivity_recall + specificity) / 2

confusion_metrics_all <- data.frame(
  method = "XGBoost_threshold_0.5",
  n = length(y_valid),
  TP = TP,
  TN = TN,
  FP = FP,
  FN = FN,
  accuracy = accuracy,
  sensitivity_recall = sensitivity_recall,
  specificity = specificity,
  precision_ppv = precision_ppv,
  npv = npv,
  f1 = f1,
  balanced_accuracy = balanced_accuracy
)

cat("\n--- 完整混淆矩阵指标 ---\n")
print(confusion_metrics_all)