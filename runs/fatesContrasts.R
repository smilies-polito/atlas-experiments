library(readr) # 2.1.5
library(tibble) #v 3.2.1
library(dplyr) #v 1.1.4
library(multcomp) #v 1.4.26


path <- "/Users/lrcq/Documents/devtraj/scvemo/palantir_results/fates.tsv"
fates <- as.data.frame(read_tsv(path))

celltype <- "RG, Astro, OPC"#insert cell type of interest for subsetting
subset.df <- fates %>% filter(`rna:celltype` == celltype)
subset.df <- subset.df %>% select(- `rna:celltype`)

subset.df$model <- factor(subset.df$model, levels = c("rna", "multiomics"))
subset.df$terminal <- factor(subset.df$terminal)

model <- lm(probability ~ model*terminal, data=subset.df)
summary(model)

contrast_matrix <- rbind(
  "Deeper Layer" = c(0,-1,0,0,0,0),
  "Upper Layer" = c(0,-1,-1,1,0,-1),
  "RG, Astro, OPC" = c(0,-1,0,0,-1,0)
)
contrast_model <- glht(model, linfct= contrast_matrix)
summary(contrast_model)


subset.df$interaction<- with(subset.df, interaction(model, terminal, sep=":"))
model <- lm(probability~interaction, data=subset.df)
contrasts <- glht(model, linfct = mcp(interaction="Tukey"))
summary(contrasts, test= adjusted("bonferroni"))
