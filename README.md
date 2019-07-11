# PGBVS
Methods for An Efficient Bayesian Varying-Coefficient Modeling Approach for Behavioral mHealth Data  

In this folder you'll find code for the R package PGBVS for the manuscript:
“An Efficient Bayesian Varying-Coefficient Modeling Approach for Behavioral mHealth Data” 
The main functions operate in C++ via the R package Rcpp. These functions can be sourced by a set of wrapper functions that enable easy implementation of the code in the R environment. 
Various functions are available that produce, summarize, and plot the results for inference.  
This package relies on various R packages that need to be installed in the R environment before running. 
To install, use the install.packages("") command for the following packages:  

  Rcpp   
    RcppArmadillo  
  sdols  
  mcclust  
  glmnet  
  spikeSlabGAM  
  MCMCpack  
  mvtnorm  
  fastDummies  
  rpql  

To install the ‘PGBVS’ package, run  
  install.packages("devtools")  
  library( devtools )  
Download the ‘PGBVS’ folder from https://github.com/mkoslovsky/PGBVS and set the working directory in Rstudio to its path.   
  setwd("<your path>")  
Then run:  
  devtools::build(vignettes = F)  
and  
  devtools::install()  
A worked example and technical details are provided in the vignette.  

