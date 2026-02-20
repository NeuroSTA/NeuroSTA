# required packages (install & load)
install.packages(c("qgraph", "bootnet", "readxl"))
install.packages("readxl")
library(qgraph)
library(bootnet)
library(readxl)

# input
file_path <- "path input (here applicable for excel-table)"  
sheet_name <- "sheet name"       # if excel-table contains more than 1 sheet 

# reading following range of matrix: from column X to column XX
data_matrix <- read_excel(file_path, sheet = sheet_name, range = "X:XX")

# check for numeric numbers
data_matrix <- as.data.frame(lapply(data_matrix, as.numeric))

# specify: -99 = NA
data_matrix[data_matrix == -99] <- NA

# network analysis estimation using EBICglasso
network <- estimateNetwork(
  data_matrix,
  default = "EBICglasso",
  corMethod = "cor_auto",  
  tuning = 0.5,             # tuning parameter 
  missing = "pairwise"     
)

# network visualisation
qgraph(network$graph, layout = "spring", theme = "classic")

# centrality measures calculation
centrality <- centralityTable(network)
print(centrality)

# bootstrapping (1000 permutations)
boot_results <- bootnet(
  network,
  nBoots = 1000,
  statistics = c("strength", "closeness", "betweenness"),
  type = "nonparametric"
)

# visualisation of bootstrap
plot(boot_results, "strength")
plot(boot_results, "closeness")
plot(boot_results, "betweenness")

# centrality measures
centrality_measures <- centrality(network)

names(centrality_measures)

# output of centrality measures 
centrality_table <- list(Node = names(network$graph))

