library(readr) # 2.1.5
library(tibble) #v 3.2.1
library(dplyr) #v 1.1.4

path <- ...
data <- as.data.frame(read_tsv(path))

data$model <- factor(data$model, levels = c("rna", "multiomics"))

celltype <- "Subplate"
subdf <- data %>% filter(`rna:celltype`==celltype)

model <- lm(entropy~model, data = subdf)
summary(model)
