library(readxl)
library(dplyr)
library(writexl)

file_in_path <- "/Users/huanqiwu/Desktop/PIL_Fullset_wID_wZ.xlsx"
out_dir <- "/Users/huanqiwu/Desktop/SetofData"

table_dir <- file.path(out_dir, "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

operation_z <- read_excel(file_in_path)


id_vars <- c("sample_id")


numeric_raw_vars <- c(
  "X2", "X8", "X28", "X59",
  "X6", "X9", "X16", "X17", "X18", "X19",
  "X20", "X21", "X22", "X23", "X25",
  "X29", "X33", "X34", "X35", "X36",
  "X37", "X38", "X41", "X42", "X43"
)


numeric_z_vars <- paste0(numeric_raw_vars, "_z")

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

outcome_var <- "Z3"

model_vars <- c(
  id_vars,
  outcome_var,
  binary_vars,
  multiclass_vars,
  numeric_z_vars
)


missing_model_vars <- setdiff(model_vars, names(operation_z))

model_data <- operation_z %>%
  dplyr::select(dplyr::any_of(model_vars))


model_excel_path <- file.path(table_dir, "PIL_model_data_z_only.xlsx")

write_xlsx(model_data, model_excel_path)

cat("done\n")
cat("path", model_excel_path, "\n")






set.seed(123)

split_dir <- file.path(out_dir, "split_data")
dir.create(split_dir, recursive = TRUE, showWarnings = FALSE)

unique_ids <- unique(model_data$sample_id)
n_ids <- length(unique_ids)

shuffled_ids <- sample(unique_ids)

cut60 <- floor(0.6 * n_ids)
cut80 <- floor(0.8 * n_ids)

id_groups <- tibble(
  sample_id = shuffled_ids
) %>%
  mutate(
    dataset = case_when(
      row_number() <= cut60 ~ "part1_60",
      row_number() <= cut80 ~ "part2_20",
      TRUE ~ "part3_20"
    )
  )

model_data_split <- model_data %>%
  inner_join(id_groups, by = "sample_id")

part1_60 <- model_data_split %>%
  filter(dataset == "part1_60") %>%
  dplyr::select(-dataset)

part2_20 <- model_data_split %>%
  filter(dataset == "part2_20") %>%
  dplyr::select(-dataset)

part3_20 <- model_data_split %>%
  filter(dataset == "part3_20") %>%
  dplyr::select(-dataset)

write_xlsx(part1_60, file.path(split_dir, "TrainingSet.xlsx"))
write_xlsx(part2_20, file.path(split_dir, "ValidSet.xlsx"))
write_xlsx(part3_20, file.path(split_dir, "TestSet.xlsx"))


count_id1 <- length(unique(part1_60$sample_id))
count_id2 <- length(unique(part2_20$sample_id))
count_id3 <- length(unique(part3_20$sample_id))

total_id <- length(unique(model_data$sample_id))

message("ID distribution：", count_id1, " : ", count_id2, " : ", count_id3)

message(
  "Percentage",
  round(count_id1 / total_id * 100, 1), "% | ",
  round(count_id2 / total_id * 100, 1), "% | ",
  round(count_id3 / total_id * 100, 1), "%"
)