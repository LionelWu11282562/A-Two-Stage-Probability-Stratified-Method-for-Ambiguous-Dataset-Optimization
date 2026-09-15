############################################
# 普通监督学习版本：
# 训练集 / 测试集均直接使用 Z4：
# Z4 = 1 为阳性，Z4 = 0 为阴性
# 同时比较：
# 1. XGBoost
# 2. Logistic Regression
# 3. Neural Network
# 输出完整混淆矩阵指标
############################################

library(xgboost)
library(tidyverse)
library(caret)
library(readxl)
library(nnet)

############################################
# 1. 输入训练集和测试集路径
############################################

train_file_path <- "/Users/huanqiwu/Desktop/FTrainF.xlsx"
test_file_path  <- "/Users/huanqiwu/Desktop/FTestF.xlsx"

train_df <- read_excel(train_file_path)
test_df  <- read_excel(test_file_path)

############################################
# 2. 定义变量
############################################

id_vars <- c("sample_id")

numeric_z_vars <- c(
  "X2_z", "X8_z", "X28_z", "X59_z",
  "X6_z", "X9_z", "X16_z", "X17_z", "X18_z", "X19_z",
  "X20_z", "X21_z", "X22_z", "X23_z", "X25_z",
  "X29_z", "X33_z", "X34_z", "X35_z", "X36_z",
  "X37_z", "X38_z", "X41_z", "X42_z", "X43_z"
)

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

all_feature_vars <- c(
  binary_vars,
  numeric_z_vars,
  multiclass_vars
)

############################################
# 3. 数据清理：直接使用 Z4 作为目标变量
############################################

train_clean <- train_df %>%
  filter(.data[[outcome_var]] %in% c(0, 1)) %>%
  mutate(target = as.integer(.data[[outcome_var]]))

test_clean <- test_df %>%
  filter(.data[[outcome_var]] %in% c(0, 1)) %>%
  mutate(target = as.integer(.data[[outcome_var]]))

############################################
# 4. 检查变量是否齐全
############################################

missing_train_vars <- setdiff(c(all_feature_vars, outcome_var), names(train_clean))
missing_test_vars  <- setdiff(c(all_feature_vars, outcome_var), names(test_clean))

if(length(missing_train_vars) > 0){
  stop("训练集缺少变量：", paste(missing_train_vars, collapse = ", "))
}

if(length(missing_test_vars) > 0){
  stop("测试集缺少变量：", paste(missing_test_vars, collapse = ", "))
}

############################################
# 5. 统一多分类变量水平，保证训练集和测试集编码一致
############################################

for(v in multiclass_vars){
  all_levels <- union(
    unique(as.character(train_clean[[v]])),
    unique(as.character(test_clean[[v]]))
  )
  
  train_clean[[v]] <- factor(as.character(train_clean[[v]]), levels = all_levels)
  test_clean[[v]]  <- factor(as.character(test_clean[[v]]),  levels = all_levels)
}

############################################
# 6. One-Hot 编码
############################################

dummies <- dummyVars(
  ~ .,
  data = train_clean[, multiclass_vars],
  fullRank = FALSE
)

train_multiclass <- predict(
  dummies,
  newdata = train_clean[, multiclass_vars]
)

test_multiclass <- predict(
  dummies,
  newdata = test_clean[, multiclass_vars]
)

############################################
# 7. 合并特征矩阵
############################################

X_train <- cbind(
  train_clean[, binary_vars],
  train_clean[, numeric_z_vars],
  train_multiclass
) %>%
  as.matrix()

X_test <- cbind(
  test_clean[, binary_vars],
  test_clean[, numeric_z_vars],
  test_multiclass
) %>%
  as.matrix()

y_train <- train_clean$target
y_test  <- test_clean$target

############################################
# 8. 自定义函数：输出完整混淆矩阵指标
############################################

calculate_metrics <- function(y_true, y_pred, method_name){
  
  TP <- sum(y_true == 1 & y_pred == 1)
  TN <- sum(y_true == 0 & y_pred == 0)
  FP <- sum(y_true == 0 & y_pred == 1)
  FN <- sum(y_true == 1 & y_pred == 0)
  
  accuracy <- (TP + TN) / (TP + TN + FP + FN)
  sensitivity_recall <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  precision_ppv <- TP / (TP + FP)
  npv <- TN / (TN + FN)
  f1 <- 2 * precision_ppv * sensitivity_recall / 
    (precision_ppv + sensitivity_recall)
  balanced_accuracy <- (sensitivity_recall + specificity) / 2
  
  metrics <- tibble(
    method = method_name,
    n = length(y_true),
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
  
  return(metrics)
}

############################################
# 9. 模型 1：XGBoost
############################################

dtrain <- xgb.DMatrix(data = X_train, label = y_train)
dtest  <- xgb.DMatrix(data = X_test,  label = y_test)

xgb_params <- list(
  objective = "binary:logistic",
  max_depth = 6,
  eta = 0.1,
  nthread = 2,
  eval_metric = "logloss"
)

set.seed(42)

xgb_model <- xgb.train(
  params = xgb_params,
  data = dtrain,
  nrounds = 100,
  verbose = 0
)

xgb_pred_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_pred_prob >= 0.5, 1, 0)

xgb_metrics <- calculate_metrics(
  y_true = y_test,
  y_pred = xgb_pred,
  method_name = "XGBoost_threshold_0.5"
)

############################################
# 10. 模型 2：Logistic Regression
############################################

train_lr_df <- as.data.frame(X_train)
train_lr_df$target <- y_train

test_lr_df <- as.data.frame(X_test)

logit_model <- glm(
  target ~ .,
  data = train_lr_df,
  family = binomial(link = "logit")
)

logit_pred_prob <- predict(
  logit_model,
  newdata = test_lr_df,
  type = "response"
)

logit_pred <- ifelse(logit_pred_prob >= 0.5, 1, 0)

logit_metrics <- calculate_metrics(
  y_true = y_test,
  y_pred = logit_pred,
  method_name = "Logistic_regression_threshold_0.5"
)

############################################
# 11. 模型 3：Neural Network
############################################

# nnet 需要 target 作为 factor 更稳妥
train_nn_df <- as.data.frame(X_train)
train_nn_df$target <- factor(y_train, levels = c(0, 1))

test_nn_df <- as.data.frame(X_test)

set.seed(42)

nn_model <- nnet(
  target ~ .,
  data = train_nn_df,
  size = 5,       # 隐藏层神经元数，可后续调参
  decay = 0.01,   # L2 正则化，防止过拟合
  maxit = 500,
  trace = FALSE
)

nn_pred_prob <- predict(
  nn_model,
  newdata = test_nn_df,
  type = "raw"
)

# 若输出为矩阵，则取类别 1 的概率；若为向量，则直接使用
# nnet 二分类时通常只输出 1 列，即预测为 1 的概率
if(is.matrix(nn_pred_prob)){
  nn_pred_prob_1 <- as.numeric(nn_pred_prob[, 1])
} else {
  nn_pred_prob_1 <- as.numeric(nn_pred_prob)
}

nn_pred <- ifelse(nn_pred_prob_1 >= 0.5, 1, 0)

nn_metrics <- calculate_metrics(
  y_true = y_test,
  y_pred = nn_pred,
  method_name = "Neural_network_threshold_0.5"
)

############################################
# 12. 合并并输出全部模型指标
############################################

all_model_metrics <- bind_rows(
  xgb_metrics,
  logit_metrics,
  nn_metrics
)

cat("\n训练集标签分布：\n")
print(table(y_train))

cat("\n测试集标签分布：\n")
print(table(y_test))

cat("\n三个模型的混淆矩阵指标：\n")
print(all_model_metrics)

############################################
# 13. 可选：分别输出 caret 混淆矩阵
############################################

cat("\n--- XGBoost 混淆矩阵 ---\n")
print(
  confusionMatrix(
    data = factor(xgb_pred, levels = c(0, 1)),
    reference = factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
)

cat("\n--- Logistic Regression 混淆矩阵 ---\n")
print(
  confusionMatrix(
    data = factor(logit_pred, levels = c(0, 1)),
    reference = factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
)

cat("\n--- Neural Network 混淆矩阵 ---\n")
print(
  confusionMatrix(
    data = factor(nn_pred, levels = c(0, 1)),
    reference = factor(y_test, levels = c(0, 1)),
    positive = "1"
  )
)

############################################
# 14. XGBoost 特征重要性
############################################

importance <- xgb.importance(
  feature_names = colnames(X_train),
  model = xgb_model
)

cat("\nXGBoost 前 10 个重要特征：\n")
print(head(importance, 10))

xgb.plot.importance(
  importance_matrix = importance[1:15, ]
)
