# ====================================================================================
# NLP Linguistic Features: Exploratory Factor Analysis Pipeline
# ====================================================================================

# ----------------------------------
# SETUP
# ----------------------------------
library(tidyverse)
library(psych)
library(GPArotation)
library(corrplot)
library(lavaan)
library(ggplot2)
library(nFactors)
library(readxl)
library(writexl)
library(summarytools)
library(rempsyc)
library(nortest)
library(MVN)
library(caret)
library(car)
library(EFAtools)
library(semPlot)
library(semTools)
library(reshape2)

set.seed(42)  


# ----------------------------------
# 1. LOAD DATA
# ----------------------------------
data_all <- read_excel('data_nlp_speech.xlsx')

#summarytools::view(dfSummary(data_all))

var <- c("TTR", "MTLD", "PRON_Ratio", "Pers_PRON_Ratio", 
         "Morph_Complexity", "Syn_complexity", "Subordination_Ratio", 
         "Readability", "Semantic_Coherence", "Sentence_Level_Coh", "Word_Level_Coh", "Semantic_Density", 
         "Connective_Ratio", "Graph_based_Cohesion",
         "Filled_Pauses", "Repetitions", "Grammatical_Error", "Negative_Sentiment")

data <- subset(data_all, select = var)

# ----------------------------------
# 2. DATA PREPROCESSING
# ----------------------------------
# Handle missing data
missing_values <- sum(is.na(data))
print(paste("Total missing values:", missing_values))

# using imputation vs. removing rows
df <- na.omit(data)

# Check for Outliers
mahalanobis_distances <- mahalanobis(df, colMeans(df), cov(df))
cutoff <- qchisq(0.95, df = ncol(df))
outliers <- mahalanobis_distances > cutoff
print(sum(outliers))

# Remove outliers if necessary
#df_cleaned <- df[!outliers, ]

# Check for Normality
univariate_normality <- apply(df, 2, function(x) ad.test(x)$p.value)
print(univariate_normality)

# Multivariate Normality
mvn_result <- mvn(df, mvnTest = "mardia")
print(mvn_result$multivariateNormality)

scaled_data <- scale(df)

# ----------------------------------
# 3. ASSUMPTION TESTS FOR FACTOR ANALYSIS
# ----------------------------------
# Correlation Matrix
cor_matrix <- cor(scaled_data)
# Plot Correlation Matrix
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8, tl.col = "black", addCoef.col = "black", number.cex = 0.7)

# Check for multicollinearity and remove highly correlated pairs
high_corr_pairs <- which(abs(cor_matrix) > 0.80 & lower.tri(cor_matrix), arr.ind = TRUE)

# Multicollinearity check
vif_data <- lm(as.matrix(scaled_data) ~ 1)
vif_results <- apply(scaled_data, 2, function(x) vif(lm(x ~ ., data = as.data.frame(scaled_data))))
print(vif_results)

# Bartlett's test & KMO
BARTLETT(scaled_data, use = "complete.obs", cor_method = "spearman")
KMO(scaled_data, use = "complete.obs", cor_method = "spearman")

# ----------------------------------
# 4. FACTOR RETENTION DECISION
# ----------------------------------
N_FACTORS(scaled_data, criteria = c("EKC", "HULL", "KGC", "PARALLEL", "SCREE"), 
          method = "ULS", use = "complete.obs", cor_method = "spearman",  eigen_type_HULL = "EFA", eigen_type_other = "EFA")

# Calculate eigenvalues
eigenvalues <- eigen(cor(scaled_data))$values

# Create a scree plot using base R
plot(eigenvalues, type = "b", pch = 19, 
     xlab = "Factor Number", 
     ylab = "Eigenvalue", 
     main = "Scree Plot of NLP Speech Features")
abline(h = 1, col = "red", lty = 2)


# ----------------------------------
# 5. EXPLORATORY FACTOR ANALYSIS
# ----------------------------------
num_factors = 3
efa_results_3f <- list(
  simplimax = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "simplimax", use = "complete.obs", cor_method = "spearman"),
  promax = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "promax", use = "complete.obs", cor_method = "spearman"),
  oblimin = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "oblimin", use = "complete.obs", cor_method = "spearman"),
  quartimin = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "quartimin", use = "complete.obs", cor_method = "spearman"),
  bentlerQ = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "bentlerQ", use = "complete.obs", cor_method = "spearman"),
  geominQ = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "geominQ", use = "complete.obs", cor_method = "spearman"),
  bifactorQ = EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "bifactorQ", use = "complete.obs", cor_method = "spearman")
)

print(efa_results_3f)

# Comparison of different methods - different lib
COMPARE(
  EFA(scaled_data, n_factors = num_factors, rotation = "promax", type = "psych")$rot_loadings,
  EFA(scaled_data, n_factors = num_factors, rotation = "promax", type = "EFAtools")$rot_loadings
)

# Average solution across many different EFAs with oblique rotations
EFA_AV <- EFA_AVERAGE(scaled_data, n_factors = num_factors, 
                      method = c("PAF", "ML", "ULS"), rotation = "oblique",
                      show_progress = FALSE)
EFA_AV

# Perform EFA with EFAtools
efa_results <- EFA(scaled_data, n_factors = num_factors, method = "ULS", rotation = "promax", use = "complete.obs", cor_method = "spearman")
print(efa_results)

# Bootstrapping
n_factors_nlp <- 3
efa_nlp_results <- fa(scaled_data, nfactors = n_factors_nlp, 
                      rotate = "promax", fm = "minres", 
                      n.iter = 5000)  # 5000 bootstrap iterations

print(efa_nlp_results, digits = 3)

# ----------------------------------
# 6. SAVE FACTOR SCORES 
# ----------------------------------
nlp_scores_df <- data.frame(Proband = data_all$Proband, efa_nlp_results$scores)
colnames(nlp_scores_df) <- c("Proband", paste0("NLP_F", 1:num_factors))

combined_data <- data_all %>% left_join(nlp_scores_df, by = "Proband")
# write_xlsx(combined_data, "data_linguistic.xlsx"


# ----------------------------------
# 7. VISUALIZATIONS OF FACTOR LOADINGS
# ----------------------------------
# Extract loadings
loadings_matrix <- as.data.frame(efa_nlp_results$loadings[, 1:num_factors])
loadings_matrix$Variable <- rownames(loadings_matrix)

# Reshape for ggplot2
library(tidyr)
colnames(loadings_matrix)[1:num_factors] <- paste0("Factor", 1:num_factors)
loadings_long <- pivot_longer(loadings_matrix, 
                              cols = starts_with("Factor"), 
                              names_to = "Factor", 
                              values_to = "Loading")


# Heatmap of Loadings
ggplot(loadings_long, aes(x = Factor, y = Variable, fill = Loading)) +
  geom_tile() +
  scale_fill_gradient2(low = "red", mid = "white", high = "blue", midpoint = 0, limits = c(-1,1)) +
  theme_minimal(base_size = 14) +
  labs(title = "Factor Loadings Heatmap", x = "Factor", y = "Variable")

# Bar Plot per factor
ggplot(loadings_long, aes(x = reorder(Variable, Loading), y = Loading, fill = Factor)) +
  geom_bar(stat = "identity", position = "dodge") +
  coord_flip() +
  theme_minimal(base_size = 14) +
  labs(title = "Standardized Factor Loadings", x = "Variables", y = "Loading") +
  scale_fill_brewer(palette = "Set1")

# Diagram of factor structure
psych::fa.diagram(efa_nlp_results, main = "Factor Structure Diagram")
