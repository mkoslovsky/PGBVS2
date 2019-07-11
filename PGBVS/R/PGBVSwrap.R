########################################################################################################################################
#' This is an R wrapper for the C++ code PGBVS.cpp that Implements Bayesian variables selection for random effects as well as varying-coefficients
#'  using spiked Dirichlet processes for logistic regression models with repeated measures data using Polya-Gamma augmentation for
#'  efficient sampling. 
#'
#'
#' Author: Matt Koslovsky 2019
########################################################################################################################################

# Simulate data 
data_sim <- function(
  N = 100,
  n_i = rep(20, 100),
  P = 10,
  D = 10,
  cor = 0.0,
  beta_bar = NULL,
  non_linear = NULL,
  kappa = NULL,
  Gamma = NULL,
  zeta = NULL, 
  seed = 121113
){
  set.seed( seed )
 
  # Get libraries to simulate data
  library( mvtnorm )
  library( MCMCpack )
  
  # Adjust the number of observations per person if necessary
  if( length( n_i ) == 1 ){ n_i <- rep( n_i, N ) }
  
  # Make X - Correlation between covariates is cor
  sigma2 <- diag( P )
  
  for( i in 1:P ){
    for( j in 1:P ){
      if( i != j ){
        sigma2[ i , j ] = cor^abs(i - j)
      }
    }
  }
  
  # Simulate a covariate for each subject
  X <- rmvnorm( N, rep( 0, nrow( sigma2 ) ), sigma2 )
  
  # Replicate some and make the others wander
  X_ij <- numeric()
  cols <- sample( seq(1,P), floor( P/2 ) )
  for( i in 1:N ){
    # Replicate by number of observations
    x_ij <- matrix( rep( X[ i, ], n_i[ i ] ), ncol = P, byrow = TRUE )
    
    # Give 1/2 of the columns random noise
    x_ij[, cols ] <- x_ij[, cols ] + matrix( rnorm( length( cols )*n_i[ i ] ), ncol = length( cols ), byrow = TRUE )
    
    # Append observations
    X_ij <- rbind( X_ij, x_ij )
  }
  
  # Add intercept term to X_ij
  X_ij <- scale( X_ij )
  
  # Copy fixed effects for random effects
  X_ij[,1] <- 1
  Z_ij <- X_ij
  
  # Make U - Assumes that the input into spline function is the same
  U <- numeric()
  for( j in 1:length( n_i ) ){
    U <- c( U, sort( runif( n_i[ j ], 0, 1 )))
  }
  U <- matrix( rep( U, ncol( X_ij) ), ncol = ncol( X_ij) )
  
  # Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
  Xbar <- numeric()
  for( p in 1:ncol( U ) ){
    tmp <- X_ij[ , p]*U[ , p ]
    Xbar <- cbind( Xbar, tmp, X_ij[ , p ] )
  }
  
  ###### Simulate data ######
  # Make home for eta
    eta <- rep( 0 , sum( n_i ) )
  
  ### Fixed effects ###
   # main effects and linear interactions 
     beta_bar. <- if( is.null( beta_bar ) ){ c( 1, -1, sample( c( 0,1.5,-1.5,2,-2), ( 2*P - 2 ), TRUE, c(.6,.1,.1,.1,.1) ) ) }else{ beta_bar } 
    
   # non-linear interactions
     non_linear. <- if( is.null( non_linear ) ){ c( 1, sample( 2:P, 2 ) ) }else{ non_linear } 
     
     for( p in non_linear. ){
       if( p == 1 ){
        eta <- eta + ( pi*sin( 3*pi*U[,p] ) )*X_ij[ , p ]
       }
       if( p == 2 ){
        eta <- eta + ( pi*cos( 2*pi*U[,p] ) )*X_ij[ , p ]
       }
       if( p == 3 ){
         eta <- eta - ( pi*U[,p]*cos( 5*pi*U[,p] ) )*X_ij[ , p ]
       }
       if( p > 3 ){
         eta <- eta - ( pi*sin( 5*pi*U[,p] ) )*X_ij[ , p ]
       }
     }
  
     eta <- eta + Xbar%*%beta_bar.
  
  ### Random effects ###
  # Matrix of random effects
  kappa. <- if( is.null( kappa ) ){ diag( c( 1, sample( c( 0,1.5,2), ( D - 1 ), TRUE, c(.6,.2,.2) ) ) ) }else{ kappa } 
  Gamma.<- if( is.null( Gamma ) ){ diag( D ) }else{ Gamma } 
  zeta. <- if( is.null( zeta ) ){ matrix( rnorm( D*N ), nrow = N, ncol = D ) }else{ zeta } 
  
  subject <- rep( seq( 0, ( length( n_i ) - 1 ) ), n_i )
  for( i in 1:sum(n_i) ){
    sub <- subject[ i ] + 1
    eta[ i ] <- eta[ i ] + Z_ij[ i, ]%*%kappa.%*%Gamma.%*%zeta.[sub, ]
  }

  prob <- exp( eta )/(1 + exp( eta) ) 
  Y <- rbinom( length(prob) , 1 , prob )
  
  data_sim <- list( "Y" = Y, "n_i" = n_i, "X" = X_ij[, -1], "U" = U, "Z" = Z_ij[,-1], "beta_bar_true" = beta_bar., "kappa_true" = kappa., "Gamma_true" = Gamma., "zeta_true" = zeta. )
  return( data_sim ) 

}

# Calculates Bayesian false discovery rate for a given threshold and also supplies the median model results 
selection <- function(mppi_vector, bfd_threshold = 0.1){
  # Courtesy of D.Wadsworth
  # arg checking
  if(any(mppi_vector > 1 | mppi_vector < 0)){
    stop("Bad input: mppi_vector should contain probabilities")
  }
  if(bfd_threshold > 1 | bfd_threshold < 0){
    stop("Bad input: bfd_threshold should be a probability")
  }
  # sorting the ppi's in decresing order
  sorted_mppi_vector <- sort(mppi_vector, decreasing = TRUE)
  # computing the fdr 
  fdr <- cumsum((1 - sorted_mppi_vector))/seq(1:length(mppi_vector))
  # determine index of the largest fdr less than bfd_threshold
  thecut.index <- max(which(fdr < bfd_threshold))
  ppi_threshold <- sorted_mppi_vector[thecut.index]
  bfd_selected <- mppi_vector > ppi_threshold
  med_selected <- mppi_vector > 0.50
  group_bfd <- floor((which( bfd_selected )  - 1)/3) + 1
  group_med <- floor((which( med_selected ) - 1)/3) + 1
  return(list(bfd_selected = bfd_selected, bfd_threshold = ppi_threshold, group_bfd = group_bfd, med_selected = med_selected, group_med = group_med ) )
}

# Provides results for Sensitivity, Specificity, and MCC for simulated data 
# Make sure that the indicies are aligned 
select_perf <- function( selected, truth ){
  
  if( any( ! selected %in% c( 0, 1 ) ) ) {
    stop("Bad input: selected should be zero or one")
  }
  if( any( ! truth %in% c( 0, 1 ) ) ) {
    stop("Bad input: truth should be zero or one")
  }
  select <- which( selected == 1 )
  not_selected <- which( selected == 0 )
  included <- which( truth == 1 )
  excluded <- which( truth == 0 )
  
  TP <- sum( select %in% included )
  TN <- sum( not_selected %in% excluded )
  FP <- sum( select %in% excluded )
  FN <- sum( not_selected %in% included )
  sensitivity <- TP/( FN + TP )
  specificity <- TN/( FP + TN ) 
  mcc <- ( TP*TN - FP*FN )/(sqrt( TP + FP )*sqrt(TP + FN )*sqrt(TN + FP )*sqrt(TN + FN) )
  
  return( list( sens = sensitivity, spec = specificity, mcc = mcc ) ) 
}

# Evaluate clustering and compare performance for simulated data if provided 
clustering <- function( MCMC = NULL, true_clusters = NULL, loss = "lowerBoundVariationOfInformation" ){
  
  if( is.null( MCMC ) ){
    stop("Missing input: Please provide a matrix of samples.")
  }
  
  if( !is.character( loss ) ){
    stop("Missing input: Please provide a loss permitted by the 'salso' function.")
  }
  
  library(sdols)
  library(mcclust)
  
  # Get clusters 
  probabilities <- expectedPairwiseAllocationMatrix( MCMC )
  cluster <- salso( probabilities, loss = loss )
  
  # Return results
  if( !is.null( true_clusters ) ){
    eval <- vi.dist( true_clusters, cluster )
    return( list( cluster = c( cluster ), eval_vi = eval ) )
  }else{
    return( list( cluster = c( cluster ) ) )
  }
  
}

# Prints varying coefficients selected by PGBVS
plot.PGBVS <- function( bvsPG.out = NULL, s.function = NULL, xlab = NULL, ylab = NULL, main = NULL, exp = FALSE  ){
  if( is.null( bvsPG.out ) ){
    stop("Missing input: Please provide a PGBVS output object.")
  }
  if( is.null( s.function ) ){
    stop("Missing input: Please provide a smooth function to plot.")
  }
  if( is.null( xlab ) ){
    xlab <- "U"
  }
  if( is.null( ylab ) & exp ){
    ylab <- "Odds Ratio"
  }
  if( is.null( ylab ) & !exp ){
    ylab <- "Log odds Ratio"
  }
  if( is.null( main ) ){
    main. <- paste("Smooth Function", s.function)
  }
  iterations <- ncol(bvsPG.out$mcmc$W)
  half <- ceiling(iterations/2) + 1
  
  non_lin <- (s.function - 1)*3 + 1
  lin <- (s.function - 1)*3 + 2
  main <- (s.function - 1)*3 + 3

  # Add all of the nonlinear, linear and main effects at each nonlinear point and then take 95%s
  locations <- bvsPG.out$Ustar_dems[ s.function:( s.function + 1 ) ]
  locations[1] <- locations[1] + 1
  Ustared <- bvsPG.out$Ustar[ , locations[1]:locations[2] ]
  hm <- bvsPG.out[[1]]$beta[ non_lin, which( bvsPG.out[[1]]$v[ non_lin, ] == 1)[ which( bvsPG.out[[1]]$v[ non_lin, ] == 1) >= half ] ]
  ok <- matrix( rep(hm, ncol( Ustared ) ), ncol = length( hm ), nrow = ncol( Ustared ), byrow = T)
  all <- Ustared%*%( ok*bvsPG.out[[1]]$xi[ locations[1]:locations[2], which( bvsPG.out[[1]]$v[ non_lin, ] == 1)[ which( bvsPG.out[[1]]$v[ non_lin, ] == 1 ) >= half ] ] )
  lowerNON <- apply( t( t( all ) ), 1, function( x ){ quantile( x, 0.025 ) } )
  upperNON <- apply( t( t( all) ), 1, function( x ){ quantile( x, 0.975 ) } )
  meanNON <- apply( t( t( all ) ) , 1, mean )
  
  lowerLIN <- apply( t( t( bvsPG.out[[1]]$beta[ lin, half:iterations ] ) )%*%bvsPG.out$U[ ,s.function ], 2, function(x){ quantile( x, 0.025 ) } )
  upperLIN <- apply( t( t( bvsPG.out[[1]]$beta[ lin, half:iterations ] ) ) %*%bvsPG.out$U[ ,s.function ], 2, function(x){ quantile( x, 0.975 ) } )
  meanLIN <- apply( t( t( bvsPG.out[[1]]$beta[ lin, half:iterations ] ) )%*%bvsPG.out$U[ ,s.function ], 2, mean )
   
  lowerMAIN <- apply( t( t( bvsPG.out[[1]]$beta[ main, half:iterations ] ) ), 2, function(x){ quantile( x, 0.025 ) } )
  upperMAIN <- apply( t( t( bvsPG.out[[1]]$beta[ main, half:iterations ] ) ), 2, function(x){ quantile( x, 0.975 ) } )
  meanMAIN <- apply( t( t( bvsPG.out[[1]]$beta[ main, half:iterations ] ) ), 2, mean )
  
  upper <- upperNON + upperLIN + upperMAIN
  lower <- lowerNON + lowerLIN + lowerMAIN  
  mean <- meanNON + meanLIN + meanMAIN   
  
  if( exp ){
    data <- cbind(bvsPG.out$U[ ,s.function ],  exp(mean) ,  exp(lower)  ,  exp(upper) )
    data <- data.frame( data[order(bvsPG.out$U[ ,s.function ]),] ) 
    colnames( data ) <- c("x","y","l","u")
    ggplot(aes( x = x, y = y), data = data ) + geom_line( aes( x = x, y = y), data = data) + geom_ribbon( aes(ymin=l,ymax=u),alpha=0.3) + ggtitle( main. ) + xlab( xlab ) + ylab( ylab ) + geom_hline(yintercept = 1 ,linetype=3)  +
      theme(plot.title = element_text(hjust = 0.5)) 
  }else{
    data <- cbind(bvsPG.out$U[ ,s.function ],  mean, lower, upper )
    data <- data.frame( data[order(bvsPG.out$U[ ,s.function ]),] ) 
    colnames( data ) <- c("x","y","l","u")
    ggplot(aes( x = x, y = y), data = data ) + geom_line( aes( x = x, y = y), data = data) + geom_ribbon( aes(ymin=l,ymax=u),alpha=0.3) + ggtitle( main. ) + xlab( xlab ) + ylab( ylab ) + geom_hline(yintercept = 0 ,linetype=3)  +
      theme(plot.title = element_text(hjust = 0.5)) 
  }
}

plot.MPPI <- function( bvsPG.out = NULL, threshold = 0.5 ,  burnin = NULL ){
  # bvsPG.out - output from bvsPG
  # threshold - vector for posterior probability of inclusion threshold for fixed and random effects, respectively
  # burnin - number of MCMC samples to drop before inference, default = 0
  
  library(ggplot2)
  
  if( length( threshold ) > 2 ){
    stop("Bad input: threshold must be a single number or a 2-dimensional vector")
  }
 
  if( length( threshold ) == 1 ){
    threshold <- c( threshold, threshold )
  }
  
  if( threshold[1] > 1 | threshold[1] < 0 ){
    stop("Bad input: threshold should be between 0 and 1")
  }

  if( threshold[2] > 1 | threshold[2] < 0 ){
    stop("Bad input: threshold should be between 0 and 1")
  }

   iterations <- ncol(bvsPG.out$mcmc$W)
   
   if( is.null( burnin ) ){
     burnin <- ceiling( iterations/2 ) + 1
   }
   
   if( burnin%%1 != 0){
     stop("Bad input: burn-in should be an integer")
   }
   
   if( burnin < 0 ){
     stop("Bad input: burn-in should be positive")
   }

   if( burnin > iterations ){
     stop("Bad input: burnin should be less than the number of iterations")
   }
  
  # Plots MPPI for fixed and random effects 
   y <- rowMeans( bvsPG.out$mcmc$v[,(burnin + 1):iterations ] )
   x <- seq(1,length(y))
   data <- data.frame(cbind(y,x))
   print(
     ggplot(data, aes(x, y) ) +
       geom_segment(aes(xend = x, yend = 0), size = 1 , lineend = "butt") +
       labs(x="Fixed Effect Index",
         y="MPPI") + geom_abline(slope = 0, intercept = threshold[ 1 ], linetype = "dashed"))

   y <- rowMeans( bvsPG.out$mcmc$lambda[,(burnin + 1):iterations ] )
   x <- seq(1,length(y))
   data <- data.frame(cbind(y,x))
   print(
     ggplot(data, aes(x, y) ) +
       geom_segment(aes(xend = x, yend = 0), size = 1 , lineend = "butt") +
       labs(x="Random Effect Index",
            y="MPPI") + geom_abline(slope = 0, intercept = threshold[ 2 ], linetype = "dashed"))
}

plot.selected <- function( bvsPG.out = NULL ){
  # bvsPG.out - output from bvsPG
  # Plots of number of selected indices for fixed and random effects 
    len <- length(bvsPG.out$mcmc$v[1,])
    plot( 1:len, apply( bvsPG.out$mcmc$v, 2, sum ), xlab = "MCMC Sample", ylab = "# of selected fixed terms", lty = 1, type = "l")
    plot( 1:len, apply( bvsPG.out$mcmc$lambda, 2, sum ), xlab = "MCMC Sample", ylab = "# of selected random terms", lty = 1, type = "l")
}

# Wrapper function for main PGBVS
bvsPG <- function ( 
iterations = 5000,
thin = 10,
Y = NULL,
n_i = NULL, 
W = NULL, 
X = NULL, 
U = NULL, 
fixed_avail = NULL,
Z = NULL, 
random_avail = NULL, 
prior = "BB", 
DP_beta = TRUE,
DP_kappa = TRUE,
beta = NULL, 
v = NULL, 
xi = NULL,
mu = NULL, 
t2 = NULL, 
vartheta = 2,
K = NULL,
lambda = NULL,
Gamma = NULL,
zeta = NULL, 
cluster_K = NULL, 
cluster_kappa = NULL,
sA = 2,
V_gamma = NULL ,
gamma_0 = NULL ,
m_star = 0,
v_star = 5, 
m_0 = 0,
v_0 = 5,
a = 1,
b = 1,
a_K = 1,
b_K = 1, 
a_0 = 5,
b_0 = 50,
a_vartheta = 1,
b_vartheta = 1, 
a_sA = 1,
b_sA = 1,
seed = 1212, 
warmstart = FALSE ){
  
    set.seed( seed )

    # Defense
    if( iterations%%1 != 0 | iterations <= 0){
      stop("Bad input: iterations should be a positive integer")
    }
    
    if( thin%%1 != 0 | thin < 0 ){
      stop("Bad input: thin should be a positive integer")
    }

    if(  prior != "BB"  ){
      stop("Bad input: prior is not in correct format")
    }

    if( DP_beta != TRUE & DP_beta != FALSE ){
      stop("Bad input: DP_beta should be a boolean")
    }

    if( DP_kappa != TRUE & DP_kappa != FALSE ){
     stop("Bad input: DP_kappa should be a boolean")
    }
  
    if( is.null( X ) ){
      stop("Missing input: Please provide a matrix of covariates")
    }
  
    if( is.null( Y ) ){
      stop("Missing input: Please provide a vector of outcomes")
    }
  
    if( is.null( n_i ) ){
      stop("Missing input: Please provide a vector of subject observations in order of Y")
    }
  
    if( is.null( U ) ){
      stop("Missing input: Please provide a vector/matrix for varying effects")
    }
  
    if( ( ncol( as.matrix( U ) ) != 1 ) & ( ncol( as.matrix( U ) ) != ncol( as.matrix( X ) ) + 1 ) ){
      stop("Bad input: Please provide a vector/matrix for varying effects with 1 or P columns")
    }
  
    if( ! is.null( beta ) & warmstart ){
      stop("Bad input: Please provide either warmstart or initial values for beta, not both")
    }
  
   # Call dependent libraries 
     library( spikeSlabGAM )
  
   # Pre-processing of input 
     # Vector that indicates which observations come from a subject. Elements are ij starts at 0
       subject <- rep( seq( 0, ( length( n_i ) - 1 ) ), n_i )
     # Input vector of dimensions for each subject in the data. Element is equal to the starting indicies for corresponding n. Last term is the number of observations
       subject_dems <- c( 0, cumsum( n_i ) )
  
     # Add intercept to X
       X <- cbind( 1, X )
    
     # If Z missing, set it to X
       if( is.null( Z ) ){
         Z <- X
       }else{
         Z <- cbind( 1, Z )
       }
       
     # Adjust P and D 
       P <- ncol( X )
       D <- ncol( Z )
     
     # If random_avail missing, set it to everything but intercept
     # Note this is for cpp, so index - 1
        if( is.null( random_avail ) ){
          random_avail <- seq( 1, D-1 ) 
        }
     
     # If fixed_avail missing, set it to everything but intercept
     # Note this is for cpp, so index - 1
        if( is.null( fixed_avail ) ){
          fixed_avail <- seq( 3, 3*P-1 )
        }
      
    # Ustar - Matrix of spline functions. Rows indexed by ij. Columns indexed by sum r_p over p. ( Should be a list if r_p != r_p' )
    # Ustar_dems - Input vector of dimensions for each spline function. Element is equal to starting indicies for corresponding p. Last term is number of columns. Ustar_dems[ 0 ] = 0 and length = P + 1
    # Allows for only one input for U 
      if( ncol( as.matrix( U ) ) == 1 ){
        Ustar <- numeric()
        Ustar_dems <- c( 0 )
        tmp <- sm( U )
        for( p in 1:P ){
          Ustar <- cbind( Ustar, tmp )
          Ustar_dems <- c( Ustar_dems, ( Ustar_dems[ p ] + ncol( tmp ) ) )
        }
        U <- matrix( rep( U, P ), ncol = P , nrow = length( Y ) )
      }else{
        Ustar <- numeric()
        Ustar_dems <- c( 0 )
        for( p in 1:ncol( U ) ){
           tmp <- sm( U[ , p ] )
           Ustar <- cbind( Ustar, tmp )
           Ustar_dems <- c( Ustar_dems, ( Ustar_dems[ p ] + ncol( tmp ) ) )
        }
      }
       
    # Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
      Xbar <- numeric()
      for( p in 1:P ){
        tmp <- X[ , p]*U[ , p ]
        Xbar <- cbind( Xbar, tmp, X[ , p ] )
      }
  
   # Set remaining priors if not given 
     if( is.null( V_gamma ) ){
        V_gamma <- diag( D*( D - 1 )/2 )  
     }
  
      if( is.null( gamma_0 ) ){
        gamma_0 <- rep( 0 , D*( D - 1 )/2 )  
      }
  
   # Allocate output memory
     N <- length( n_i )
     samples <- floor( iterations/thin )     
     W. <-  matrix( 0, nrow = sum( n_i ), ncol = samples )               # - Matrix of MCMC samples for auxillary variables. Rows indexed ij. Columns indexed by MCMC sample
     beta. <- matrix( 0, nrow = 3*P, ncol = samples )                    # - Matrix of MCMC samples for beta. Rows indexed by beta_temp. Columns indexed by MCMC sample.
     v. <-  matrix( 0, nrow = 3*P, ncol = samples )                      # - Matrix of MCMC samples for v. Rows indexed by v_temp. Columns indexed by MCMC sample.
     xi. <-  matrix( 0, nrow = ncol( Ustar ), ncol = samples )           # - Matrix of MCMC samples for parameter expansion for beta. Rows indexed by xi_temp. Columns indexed by MCMC sample.
     mu. <-  matrix( 0, nrow = ncol( Ustar ), ncol = samples )           # - Matrix of MCMC samples of means for each xi. Rows indexed by mu_tempp. Columns indexed by MCMC sample.
     t2. <-  matrix( 0, nrow = 3*P, ncol = samples )                     # - Matrix of MCMC samples for t2. Rows indexed by t2_temp. Columns indexed by MCMC sample.
     K. <-  matrix( 0, nrow = D, ncol = samples )                        # - Matrix of MCMC samples for K. Rows indexed by K. Columns indexed by MCMC sample.
     lambda. <-  matrix( 0, nrow = D, ncol = samples )                   # - Matrix of MCMC samples for lambda. Rows indexed by D. Columns indexed by MCMC sample
     Gamma. <- array( 0, dim = c( D, D, samples ) )                      # - Array of MCMC samples for Gamma. x and y are indexed by Gamma_temp. z indexed by MCMC sample
     zeta. <-  array( 0, dim = c( N, D, samples ) )                      # - Array of MCMC samples for zeta.  x and y indexed by zeta_temp. z indexed by MCMC sample
     cluster. <- matrix( 0, nrow = 3*P, ncol = samples )                 # - Matrix of MCMC samples for beta clusters. Rows indexed by cluster. Columns indexed by MCMC sample.
     cluster_beta. <- matrix( 0, nrow = 3*P, ncol = samples )            # - Vector of beta values associated with each cluster 
     cluster_K. <- matrix( 0, nrow = D, ncol = samples )                 # - Matrix of MCMC samples for kappa clusters. Rows indexed by cluster. Columns indexed by MCMC sample.
     cluster_kappa. <- matrix( 0, nrow = D, ncol = samples )             # - Matrix of MCMC samples for kappa values associated with each cluster 
     vartheta. <- rep( 0, samples )                                      # - Vector of MCMC samples for beta concentration parameter
     sA. <- rep( 0, samples )                                            # - Vector of MCMC samples for kappa concentration parameter 
  
   # Adjust initial values given input 
     xi.[ ,1 ] <- ifelse( is.null( xi ), 1, xi )
     mu.[ , 1] <- if( is.null( mu ) ){ ifelse( rbinom( ncol( Ustar ), 1, 0.5 ) == 1, 1, -1 )}else{ mu }
     t2.[ , 1 ] <- ifelse( is.null( t2 ), 5, t2 )
     Gamma.[ , , 1] <- if( is.null( Gamma ) ){ diag( D ) }else{ Gamma }
     zeta.[ , , 1] <- if( is.null( zeta ) ){ rnorm( N*D ) }else{ zeta }
     beta.[ , 1] <- if( is.null( beta ) ){ c( 0, 1.1, 1.2, rep( c( 0, 0, 0 ), ( P - 1 ) ) )  }else{ beta }
     
     # Adjust if warmstart 
     if( warmstart ){
       library(glmnet)
       cv_fit <- cv.glmnet( Xbar, Y, alpha = 1, family = "binomial" )
       fit <- glmnet( Xbar , Y, alpha = 1, family = "binomial", lambda = cv_fit$lambda.1se )
       beta_init <- as.vector( fit$beta ) 
       beta_init[ 2 ] <- fit$a0
       beta_full_init <- rep( 0, P*3 )
       index <- 1
       for( i in 1:length( beta_full_init ) ){
         if( i%%3 != 1 ){
           beta_full_init[ i ] <- beta_init[ index ]
           index <- index + 1 
         }
       }
       beta.[ , 1] <- beta_full_init
      }
     
     
     v.[ , 1] <- if( is.null( v ) ){ ( beta.[ , 1] != 0 )*1 }else{ v }
     # Make sure that the forced in are included
     v.[ seq(1, 3*P)[! seq(1, 3*P) %in% ( fixed_avail + 1 ) ], 1] <- 1
     
     # Initiate these to be adaptive to the inital beta to protect against improper input 
     cluster_count. <- rep( 0, P*3 )
     cluster.[ , 1 ] <- rep( -1, P*3 )
     cluster_beta.[ ,1 ] <- rep( 0, P*3 )
     
     # Collect non zero betas and set an index for them 
     beta_found <- numeric()
     beta_cluster_index <- seq(0, ( P*3 - 1 ) )
     
     for( c in 1:( P*3 ) ){
       hold <- beta.[ c, 1 ]
       if( hold != 0 ){
         if( ! hold %in% beta_found ){
           beta_found <- c( beta_found, hold )
           cluster.[ which( beta.[,1] == hold ) , 1 ] <- beta_cluster_index[ 1 ]  # Set all of the betas = hold to the new cluster 
           cluster_beta.[ ( beta_cluster_index[ 1 ] + 1 ) , 1 ] <- hold 
           cluster_count.[ beta_cluster_index[ 1 ] + 1] <- sum( beta.[,1] == hold ) 
           beta_cluster_index <- beta_cluster_index[ -1 ]  # Remove available cluster 
         }
       }
     }
     
     vartheta.[ 1 ] <- vartheta 
    
     K.[ , 1] <- if( is.null( K ) ){ c( 1 , rep( 0, D - 1 ) ) }else{ K }
     lambda.[ , 1] <-  if( is.null( lambda ) ){ ( K.[ , 1] != 0 )*1 }else{ lambda }
     # Make sure that the forced in are included
     lambda.[ seq(1, D)[! seq(1, D) %in% ( random_avail + 1 ) ], 1] <- 1
     sA.[ 1 ] <- sA
     
     
     # Initiate these to be adaptive to the inital kappa to protect against improper input 
     cluster_count_K. <- rep( 0, D )
     cluster_K.[ , 1 ] <- rep( -1, D )
     cluster_kappa.[ ,1 ] <- rep( 0, D )
     
     # Collect non zero betas and set an index for them 
     kappa_found <- numeric()
     kappa_cluster_index <- seq(0, ( D - 1 ) )
     
     for( c in 1:( D ) ){
       hold <- K.[ c , 1]
       if( hold != 0 ){
         if( ! hold %in% kappa_found ){
           kappa_found <- c( kappa_found, hold )
           cluster_K.[ which( K.[,1] == hold ) , 1 ] <- kappa_cluster_index[ 1 ]  # Set all of the kappas = hold to the new cluster 
           cluster_kappa.[ ( kappa_cluster_index[ 1 ] + 1 ) , 1 ] <- hold 
           cluster_count_K.[ kappa_cluster_index[ 1 ] + 1] <- sum( K.[,1] == hold ) 
           kappa_cluster_index <- kappa_cluster_index[ -1 ]  # Remove available cluster 
         }
       }
     }
  
 ###### RUN MODEL ######
   # State time 
     ptm <- proc.time()
 
      output <- bvsPGcpp( iterations, thin, prior, DP_beta, DP_kappa, Y, W.,
                       subject, subject_dems, Ustar, Ustar_dems, Xbar, Z, random_avail,
                       fixed_avail, beta., v., xi., mu., t2., cluster.,  cluster_count.,
                       cluster_beta., vartheta., K., lambda., Gamma., zeta., cluster_K.,
                       cluster_count_K., cluster_kappa., sA., V_gamma, gamma_0, m_star,       
                       v_star, m_0, v_0, a, b, a_K, b_K, a_0, b_0, a_vartheta, b_vartheta,          
                       a_sA, b_sA )

   # Stop the clock
     total_time <- proc.time() - ptm
     
     names( output ) <- c("W", "beta", "v", "xi", "mu", "t2", "K", "lambda", "Gamma", "zeta", "cluster", "cluster_beta", "vartheta", "cluster_K", "cluster_kappa", "sA" )
     
     # Return based on prior
     if( DP_beta & DP_kappa ){
       output <- output[ c(1,2,3,4,5,7,8,9,10,11,12,13,14,15,16) ]
     }
     if( DP_beta & !DP_kappa ){
       output <- output[ c(1,2,3,4,5,7,8,9,10,11,12,13) ]
     }
     if( !DP_beta & DP_kappa ){
       output <- output[ c(1,2,3,4,5,7,8,9,10,14,15,16) ]
     }
     if( !DP_beta & !DP_kappa ){
       output <- output[1:10]
     }
     
     return( list( mcmc = output, total_time = total_time, Ustar = Ustar, Ustar_dems = Ustar_dems, Y = Y, X = X, Z = Z, U = U ) )
}