############################################
# N / U / P Labeling Based on Logistic Probability
#
# Rule:
# N: prob_A <= C group 10th percentile
# U: C group 10th percentile < prob_A < A group 75th percentile
# P: prob_A >= A group 75th percentile
############################################

library(readxl)
library(dplyr)
library(ggplot2)
library(writexl)

############################################
# 1. 路径设置
############################################

file_in_path <- "/Users/huanqiwu/Desktop/FirstLProb/logistic_results/tables/TrainingSet_with_logistic_prob_A.xlsx"

out_dir <- "/Users/huanqiwu/Desktop/FirstLProb/PU_labeling_results"
fig_dir <- file.path(out_dir, "figures")
table_dir <- file.path(out_dir, "tables")

dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

out_file <- file.path(table_dir, "TrainingSet_with_PU_label.xlsx")

############################################
# 2. 读取数据
############################################

data_prob <- read_excel(file_in_path)

############################################
# 3. 设置变量名
############################################

group_var <- "Z3"
prob_var <- "prob_A"

if (!group_var %in% names(data_prob)) {
  stop("数据中找不到 Z3")
}

if (!prob_var %in% names(data_prob)) {
  stop("数据中找不到 prob_A")
}

############################################
# 4. 计算原始 U 区间上下界
############################################

C_q10 <- data_prob %>%
  filter(.data[[group_var]] == "C") %>%
  summarise(
    q10 = quantile(.data[[prob_var]], probs = 0.10, na.rm = TRUE)
  ) %>%
  pull(q10)

A_q75 <- data_prob %>%
  filter(.data[[group_var]] == "A") %>%
  summarise(
    q75 = quantile(.data[[prob_var]], probs = 0.75, na.rm = TRUE)
  ) %>%
  pull(q75)

if (is.na(C_q10)) {
  stop("C_q10 计算失败，请检查 C 组是否存在")
}

if (is.na(A_q75)) {
  stop("A_q75 计算失败，请检查 A 组是否存在")
}

threshold_table <- data.frame(
  threshold_name = c(
    "C_group_10th_percentile_U_lower_bound",
    "A_group_75th_percentile_U_upper_bound"
  ),
  threshold_value = c(C_q10, A_q75)
)

print(threshold_table)

############################################
# 5. 生成三标签：N / U / P
############################################

data_pu <- data_prob %>%
  mutate(
    PU_label = case_when(
      .data[[prob_var]] <= C_q10 ~ "N",
      .data[[prob_var]] > C_q10 & .data[[prob_var]] < A_q75 ~ "U",
      .data[[prob_var]] >= A_q75 ~ "P",
      TRUE ~ NA_character_
    ),
    PU_label = factor(PU_label, levels = c("N", "U", "P"))
  )

############################################
# 6. 检查标签数量
############################################

label_count <- data_pu %>%
  count(PU_label, name = "n") %>%
  mutate(
    proportion = n / sum(n)
  )

label_count_by_ABC <- data_pu %>%
  count(.data[[group_var]], PU_label, name = "n") %>%
  group_by(.data[[group_var]]) %>%
  mutate(
    within_group_proportion = n / sum(n)
  ) %>%
  ungroup()

print("N / U / P 标签总体分布：")
print(label_count)

print("N / U / P 标签按 A / B / C 分布：")
print(label_count_by_ABC)

############################################
# 7. 保存结果，覆盖原 TrainingSet_with_PU_label.xlsx
############################################

write_xlsx(
  list(
    data_with_PU_label = data_pu,
    threshold_table = threshold_table,
    NUP_count = label_count,
    NUP_count_by_ABC = label_count_by_ABC
  ),
  out_file
)

############################################
# 8. 单独保存 N / U / P 数据集
############################################

data_N <- data_pu %>% filter(PU_label == "N")
data_U <- data_pu %>% filter(PU_label == "U")
data_P <- data_pu %>% filter(PU_label == "P")

write_xlsx(data_N, file.path(table_dir, "N_only_dataset.xlsx"))
write_xlsx(data_U, file.path(table_dir, "U_only_dataset.xlsx"))
write_xlsx(data_P, file.path(table_dir, "P_only_dataset.xlsx"))

############################################
# 9. 整体概率分布图：按 N / U / P 标签区分
############################################

p_label_density <- ggplot(
  data_pu,
  aes(
    x = .data[[prob_var]],
    fill = PU_label
  )
) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = C_q10, linetype = "dashed") +
  geom_vline(xintercept = A_q75, linetype = "dashed") +
  labs(
    title = "Overall Probability Distribution by N / U / P Label",
    subtitle = "N: <= C q10; U: C q10 to A q75; P: >= A q75",
    x = "Predicted probability of A",
    y = "Density",
    fill = "Label"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "overall_probability_distribution_by_NUP_label.png"),
  plot = p_label_density,
  width = 8,
  height = 6,
  dpi = 300
)

############################################
# 10. 整体概率分布图：按原始 A / B / C 区分
############################################

p_abc_density <- ggplot(
  data_pu,
  aes(
    x = .data[[prob_var]],
    fill = .data[[group_var]]
  )
) +
  geom_density(alpha = 0.4) +
  geom_vline(xintercept = C_q10, linetype = "dashed") +
  geom_vline(xintercept = A_q75, linetype = "dashed") +
  labs(
    title = "Overall Probability Distribution by Original A / B / C Group",
    subtitle = "Dashed lines: C q10 and A q75",
    x = "Predicted probability of A",
    y = "Density",
    fill = "Original Group"
  ) +
  theme_bw()

ggsave(
  filename = file.path(fig_dir, "overall_probability_distribution_by_ABC.png"),
  plot = p_abc_density,
  width = 8,
  height = 6,
  dpi = 300
)

############################################
# 11. 完成提示
############################################

cat("完成：三标签 N / U / P 已生成。\n")
cat("C_q10 = ", C_q10, "\n", sep = "")
cat("A_q75 = ", A_q75, "\n", sep = "")
cat("N 数量：", nrow(data_N), "\n")
cat("U 数量：", nrow(data_U), "\n")
cat("P 数量：", nrow(data_P), "\n")
cat("输出文件：", out_file, "\n")