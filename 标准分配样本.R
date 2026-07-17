############################################
# 从 FTestV.xlsx 中基于 sample_id 随机抽取 8611 个样本
# 并保存为新的 xlsx 文件
############################################

library(readxl)
library(dplyr)
library(writexl)

set.seed(123)  # 保证每次抽样结果可重复

# 1. 读取原始文件
data <- read_excel("/Users/huanqiwu/Desktop/FTrainV.xlsx")

# 2. 检查 sample_id 是否唯一
stopifnot(n_distinct(data$sample_id) == nrow(data))

# 3. 随机抽取 8611 个 sample_id 对应的完整样本
sampled_data <- data %>%
  slice_sample(n = 16180)

# 4. 输出为新文件
write_xlsx(
  sampled_data,
  "/Users/huanqiwu/Desktop/FTrainF.xlsx"
)

# 5. 简单检查
cat("原始样本数：", nrow(data), "\n")
cat("抽取后样本数：", nrow(sampled_data), "\n")
cat("抽取后唯一 sample_id 数：", n_distinct(sampled_data$sample_id), "\n")