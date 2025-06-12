# Network Analysis

This folder contains a fully reproducible pipeline to conduct a network analysis as used in the associated manuscript.

## Overview
This pipeline reads structured data from an Excel sheet and performs a network estimation using the EBICglasso method. It includes preprocessing, visualization, centrality estimation, and bootstrapped stability analyses.

### File Types:
- Input: here excel (.xls) files
- Expected format: numeric variables only; missing values if encoded as -99 are recoded as NA

### Pipeline Steps:
**1.** Required R packages
install.packages(c("qgraph", "bootnet", "readxl"))

**2.** Data Input
- Input by user

**3.** Network Estimation
- Method: EBICglasso
- Correlation: cor_auto (automatic detection)
- Tuning parameter: 0.5

**4.** Visualization
- Layout: spring
- Theme: classic (via qgraph)

**5.** Centrality
- Centrality indices calculated: strength, closeness, betweenness
- Output: table with centrality measurements will be printed & saved

**6.** Bootstrapping
- Nonparametric bootstrapping with 1000 iterations
- Bootstrapped stability plots for: strength, closeness, betweenness

### Reproducibility Notes
- R version ≥ 4.1 recommended
- modify your excel range (X:XX) as needed

### References for further information:
*Tibshirani R. Regression shrinkage and selection via the lasso. J R Stat Soc Series B Stat Methodol 1996;58:267–88.*

*Chen J, Chen Z. Extended Bayesian information criteria for model selection with large model spaces. Biometrika 2008;95:759–71.*

*Foygel R, Drton M. Extended Bayesian Information Criteria for Gaussian Graphical Models. In: Lafferty J, Williams C, Shawe-Taylor J, Zemel R, Culotta A, editors. Adv Neural Inf Process Syst, vol. 23, Curran Associates, Inc.; 2010.*

*Burger J, Isvoranu AM, Lunansky G, Haslbeck JMB, Epskamp S, Hoekstra RHA, et al. Reporting Standards for Psychological Network Analyses in Cross-Sectional Data. Psychol Methods 2023;28:806–24. https://doi.org/10.1037/met0000471.*

*Epskamp S, Fried EI. A tutorial on regularized partial correlation networks. Psychol Methods 2018;23:617.*
