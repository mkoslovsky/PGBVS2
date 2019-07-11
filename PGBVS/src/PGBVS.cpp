/*  Implements Bayesian variables selection for random effects as well as varying-coefficients
 *  for logistic regression with repeated measures data using Polya Gamma Augmentation for 
 *  efficient sampling. 
 *  
 *  This version allows a DP prior for fixed main effects and linear interaction terms as
 *  well as random effects
 *  
 *  Author Matt Koslovsky 2019
 *  
 *   References and Thanks to: 
 *
 *   Jesse Bennett Windle
 *   Forecasting High-Dimensional, Time-Varying Variance-Covariance Matrices
 *   with High-Frequency Data and Sampling Polya-Gamma Random Variates for
 *   Posterior Distributions Derived from Logistic Likelihoods  
 *   PhD Thesis, 2013   
 *
 *   Damien, P. & Walker, S. G. Sampling Truncated Normal, Beta, and Gamma Densities 
 *   Journal of Computational and Graphical Statistics, 2001, 10, 206-215
 *
 *   Chung, Y.: Simulation of truncated gamma variables 
 *   Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
 *
 *   Makalic, E. & Schmidt, D. F. High-Dimensional Bayesian Regularised Regression with the BayesReg Package 
 *   arXiv:1611.06649 [stat.CO], 2016 https://arxiv.org/pdf/1611.06649.pdf 
 */

//#include <omp.h>

#include <RcppArmadillo.h>
//[[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;

// Mathematical constants
#define MATH_PI        3.141592653589793238462643383279502884197169399375105820974
#define MATH_PI_2      1.570796326794896619231321691639751442098584699687552910487
#define MATH_2_PI      0.636619772367581343075535053490057448137838582961825794990
#define MATH_PI2       9.869604401089358618834490999876151135313699407240790626413
#define MATH_PI2_2     4.934802200544679309417245499938075567656849703620395313206
#define MATH_SQRT1_2   0.707106781186547524400844362104849039284835937688474036588
#define MATH_SQRT_PI_2 1.253314137315500251207882642405522626503493370304969158314
#define MATH_LOG_PI    1.144729885849400174143427351353058711647294812915311571513
#define MATH_LOG_2_PI  -0.45158270528945486472619522989488214357179467855505631739
#define MATH_LOG_PI_2  0.451582705289454864726195229894882143571794678555056317392

namespace help{

// Generate exponential distribution random variates
double exprnd(double mu)
{
  return -mu * (double)std::log(1.0 - (double)R::runif(0.0,1.0));
}

// Function a_n(x) defined in equations (12) and (13) of
// Bayesian inference for logistic models using Polya-Gamma latent variables
// Nicholas G. Polson, James G. Scott, Jesse Windle
// arXiv:1205.0310
//
// Also found in the PhD thesis of Windle (2013) in equations
// (2.14) and (2.15), page 24
double aterm(int n, double x, double t)
{
  double f = 0;
  if(x <= t) {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) + 1.5*(MATH_LOG_2_PI- (double)std::log(x)) - 2*(n + 0.5)*(n + 0.5)/x;
  }
  else {
    f = MATH_LOG_PI + (double)std::log(n + 0.5) - x * MATH_PI2_2 * (n + 0.5)*(n + 0.5);
  }    
  return (double)exp(f);
}

// Generate inverse gaussian random variates
double randinvg(double mu)
{
  // sampling
  double u = R::rnorm(0.0,1.0);
  double V = u*u;
  double out = mu + 0.5*mu * ( mu*V - (double)std::sqrt(4.0*mu*V + mu*mu * V*V) );
  
  if(R::runif(0.0,1.0) > mu /(mu+out)) {    
    out = mu*mu / out; 
  }    
  return out;
}

// Sample truncated gamma random variates
// Ref: Chung, Y.: Simulation of truncated gamma variables 
// Korean Journal of Computational & Applied Mathematics, 1998, 5, 601-610
double truncgamma()
{
  double c = MATH_PI_2;
  double X, gX;
  
  bool done = false;
  while(!done)
  {
    X = help::exprnd(1.0) * 2.0 + c;
    gX = MATH_SQRT_PI_2 / (double)std::sqrt(X);
    
    if(R::runif(0.0,1.0) <= gX) {
      done = true;
    }
  }
  
  return X;  
}

// Sample truncated inverse Gaussian random variates
// Algorithm 4 in the Windle (2013) PhD thesis, page 129
double tinvgauss(double z, double t)
{
  double X, u;
  double mu = 1.0/z;
  
  // Pick sampler
  if(mu > t) {
    // Sampler based on truncated gamma 
    // Algorithm 3 in the Windle (2013) PhD thesis, page 128
    while(1) {
      u = R::runif(0.0, 1.0);
      X = 1.0 / help::truncgamma();
      
      if ((double)std::log(u) < (-z*z*0.5*X)) {
        break;
      }
    }
  }  
  else {
    // Rejection sampler
    X = t + 1.0;
    while(X >= t) {
      X = help::randinvg(mu);
    }
  }    
  return X;
}


// Sample PG(1,z)
// Based on Algorithm 6 in PhD thesis of Jesse Bennett Windle, 2013
// URL: https://repositories.lib.utexas.edu/bitstream/handle/2152/21842/WINDLE-DISSERTATION-2013.pdf?sequence=1
double samplepg(double z)
{
  //  PG(b, z) = 0.25 * J*(b, z/2)
  z = (double)std::fabs((double)z) * 0.5;
  
  // Point on the intersection IL = [0, 4/ log 3] and IR = [(log 3)/pi^2, \infty)
  double t = MATH_2_PI;
  
  // Compute p, q and the ratio q / (q + p)
  // (derived from scratch; derivation is not in the original paper)
  double K = z*z/2.0 + MATH_PI2/8.0;
  double logA = (double)std::log(4.0) - MATH_LOG_PI - z;
  double logK = (double)std::log(K);
  double Kt = K * t;
  double w = (double)std::sqrt(MATH_PI_2);
  
  double logf1 = logA + R::pnorm(w*(t*z - 1),0.0,1.0,1,1) + logK + Kt;
  double logf2 = logA + 2*z + R::pnorm(-w*(t*z+1),0.0,1.0,1,1) + logK + Kt;
  double p_over_q = (double)std::exp(logf1) + (double)std::exp(logf2);
  double ratio = 1.0 / (1.0 + p_over_q); 
  
  double u, X;
  
  // Main sampling loop; page 130 of the Windle PhD thesis
  while(1) 
  {
    // Step 1: Sample X ? g(x|z)
    u = R::runif(0.0,1.0);
    if(u < ratio) {
      // truncated exponential
      X = t + help::exprnd(1.0)/K;
    }
    else {
      // truncated Inverse Gaussian
      X = help::tinvgauss(z, t);
    }
    
    // Step 2: Iteratively calculate Sn(X|z), starting at S1(X|z), until U ? Sn(X|z) for an odd n or U > Sn(X|z) for an even n
    int i = 1;
    double Sn = help::aterm(0, X, t);
    double U = R::runif(0.0,1.0) * Sn;
    int asgn = -1;
    bool even = false;
    
    while(1) 
    {
      Sn = Sn + asgn * help::aterm(i, X, t);
      
      // Accept if n is odd
      if(!even && (U <= Sn)) {
        X = X * 0.25;
        return X;
      }
      
      // Return to step 1 if n is even
      if(even && (U > Sn)) {
        break;
      }
      
      even = !even;
      asgn = -asgn;
      i++;
    }
  }
  return X;
}

// Simulate MVT normal data
arma::mat mvrnormArma( int n, arma::vec mu, arma::mat sigma ) {
  int ncols = sigma.n_cols;
  arma::mat Y = arma::randn( n, ncols );
  return arma::repmat( mu, 1, n ).t()  + Y * arma::chol( sigma );
}

// Calculate mvt normal density (Not using it because it is slower than my code)
arma::vec dmvnrm_arma(arma::mat x,  
                      arma::rowvec mean,  
                      arma::mat sigma, 
                      bool logd = false) { 
  int n = x.n_rows;
  int xdim = x.n_cols;
  const double log2pi = std::log(2.0 * M_PI);
  arma::vec out(n);
  arma::mat rooti = arma::trans(arma::inv(trimatu(arma::chol(sigma))));
  double rootisum = arma::sum(log(rooti.diag()));
  double constants = -(static_cast<double>(xdim)/2.0) * log2pi;
  
  for (int i=0; i < n; i++) {
    arma::vec z = rooti * arma::trans( x.row(i) - mean) ;    
    out(i)      = constants - 0.5 * arma::sum(z%z) + rootisum;     
  }  
  
  if (logd == false) {
    out = exp(out);
  }
  return(out);
}


// Make sX from Ustar, xi, Xbar, mu, Ustar_dems 
arma::mat make_sX( arma::mat Ustar, arma::vec xi, arma::mat Xbar, arma::vec Ustar_dems ){
  int P = Ustar_dems.size() - 1;
  int obs = Ustar.n_rows;
  
  arma::mat sX( obs, 3*P );
  sX.zeros();
  
  for( int p = 0; p < P; ++p ){
    arma::vec sum_p( obs );
    sum_p.zeros();
    
    // Get the range of columns corresponding to covariate p 
    arma::mat UstarXi = Ustar.cols( Ustar_dems[ p ], Ustar_dems[ p + 1 ] - 1) ;
    
    for( int ij = 0; ij < obs ; ++ij  ){
      arma::mat UstarXi_ind = UstarXi.row( ij ) * xi.subvec( Ustar_dems[ p ], Ustar_dems[ p + 1 ] - 1 ); 
      sum_p[ ij ] +=  UstarXi_ind[ 0 ];
    }
    
    // Order the covariates 
    sX.col( 3*p ) = sum_p % Xbar.col( 2*p + 1 );
    sX.col( 3*p + 1 ) = Xbar.col( 2*p );
    sX.col( 3*p + 2 ) = Xbar.col( 2*p + 1 ); 
    
  }
  return sX; 
}

// Make starX from B*, Ustar, x
arma::mat make_Xstar( arma::mat Ustar, arma::vec beta_temp, arma::mat Xbar, arma::vec Ustar_dems ){
  int P = Ustar_dems.size() - 1;
  int obs = Ustar.n_rows;
  int Rp = Ustar_dems[ Ustar_dems.size() - 1 ];
  
  arma::mat starX( obs, Rp );
  starX.zeros();
  
  for( int p = 0; p < P; ++p ){
    for( int rp = Ustar_dems[ p ]; rp < Ustar_dems[ p + 1 ]; ++rp ){
      starX.col( rp ) = beta_temp[ 3*p ] * Ustar.col( rp ) % Xbar.col( 2*p + 1 );
    }
  }  
  return starX; 
}

// Make Zhat from Z, K, zeta
arma::mat make_Zhat( arma::mat Z, arma::vec K_temp, arma::mat zeta_temp) {
  int D = Z.n_cols;
  int obs = Z.n_rows;
  
  arma::mat Zhat( obs, D*(D-1)/2 );
  Zhat.zeros();
  
  int count = 0;
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      Zhat.col( count ) = K_temp[ l ] * Z.col( l ) % zeta_temp.col( m );
      count += 1;
    }
  }
  return Zhat; 
}
// Make Zhat from Z, K, zeta and lambda
// This only makes a Zhat if its corresponding random effect components are included in the model 
arma::mat make_Zhat_lambda( arma::mat Z, arma::vec K_temp, arma::mat zeta_temp, arma::vec lambda_temp, arma::vec subject ) {
  int D = Z.n_cols;
  int D_lambda = sum( lambda_temp );
  int obs = Z.n_rows;
  
  arma::mat Zhat( obs, D_lambda*(D_lambda-1)/2 );
  Zhat.zeros();
  
  int count = 0;
  for( int m = 0; m < D - 1; ++m ){
    
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        for( int j = 0; j < obs; ++j ){
          int sub = subject[ j ];
          Zhat( j, count ) = K_temp[ l ] * Z( j, l ) * zeta_temp( sub, m );
        }
        //Zhat.col( count ) = K_temp[ l ] * Z.col( l ) % zeta_temp.col( m );
        count += 1;
        
      }
    }
  }
  return Zhat; 
}

// Make Zstar from Z, zeta, Gamma
arma::mat make_Zstar( arma::mat Z, arma::mat zeta_temp, arma::mat Gamma_temp, arma::vec subject ) {
  int D = Z.n_cols;
  int obs = Z.n_rows;
  arma::mat Zstar( obs, D );
  Zstar.zeros();
  
  for( int i = 0; i < obs ; ++i ){
    int sub = subject[ i ];
    for( int m = 0; m < D ; ++m ){
      double sum_gamma_zeta = 0;
      for( int l = 0; l < m  ; ++l ){
        sum_gamma_zeta += Gamma_temp( m, l ) * zeta_temp( sub, l ); 
      }
      Zstar( i, m ) = Z( i, m ) * ( zeta_temp( sub, m ) + sum_gamma_zeta );
    }
  }
  
  
  // Return output 
  return Zstar; 
}


// Sample from an integrer vector 
int sample_cpp( IntegerVector x ){
  // Calling sample()
  Function f( "sample" );
  IntegerVector sampled = f( x, Named( "size" ) = 1 );
  return sampled[ 0 ];
}

// Function :: Log-likelihood: h_ij = k_ij/w_ij and h ~ N( psi, Omega), k_ij = y_ij - 1/2
double log_like_pg( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject)
{
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  double log_like = 0;
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make psi and calculate the log-likelihood contribution 
  for( int i = 0; i < obs; ++i ){
    int sub = subject[ i ];
    arma::mat psi_val = sX.row( i )*beta_temp + Z.row( i )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() ; 
    double Winv = 1/W[i];
    log_like += -0.50*log( 2*M_PI*Winv ) - 1/( 2*Winv )*pow( H[i] - psi_val[0], 2 );
  }
  
  // Return output
  return log_like;
}

// Function :: likelihood for beta clusters for different values of beta_c

arma::vec like_pg_beta_clust( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, NumericVector non_zero_betas, int S_index, int non_zero_clusters )
{
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  arma::vec like( non_zero_clusters );
  like.zeros();
  
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make psi and calculate the log-likelihood contribution 
  arma::mat psi_val( non_zero_clusters, 1 );
  
  for( int i = 0; i < obs; ++i ){
    int sub = subject[ i ];
    arma::mat psi_random = Z.row( i )*K_mat*Gamma_temp*zeta_temp.row( sub ).t();
    arma::vec beta_adj = beta_temp;
    
    // Make likelihood for each of the potential cluster values 
    for( int j = 0; j < non_zero_clusters; ++j ){
      beta_adj[ S_index ] = non_zero_betas[ j ];
      arma::mat psi_fixed = sX.row( i )*beta_adj;
      psi_val( j, 0 ) = psi_fixed[ 0 ] + psi_random[ 0 ];
    }
    
    double Winv = 1/W[i];
    like += -0.50*log( 2*M_PI*Winv ) - 1/( 2*Winv )*pow( H[i] - psi_val, 2 );
  }
  
  // Return output
  return like;
}

// Function :: likelihood for kappa clusters for different values of kappa_d
arma::vec like_pg_kappa_clust( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, NumericVector non_zero_kappas, int D_index, int non_zero_clusters_K )
{
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  arma::vec H( obs );
  H.zeros();
  arma::vec like( non_zero_clusters_K );
  like.zeros();
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Get Z*
  arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
  
  // Make psi and calculate the log-likelihood contribution 
  arma::mat psi_val( non_zero_clusters_K, 1 );
  
  for( int i = 0; i < obs; ++i ){
    arma::mat psi_fixed = sX.row( i )*beta_temp;
    arma::vec kappa_adj = K_temp;
    
    // Make likelihood for each of the potential cluster values 
    for( int d = 0; d < non_zero_clusters_K; ++d ){
      kappa_adj[ D_index ] = non_zero_kappas[ d ];
      arma::mat psi_random = Zstar.row( i )*kappa_adj;   
      psi_val( d, 0 ) = psi_fixed[ 0 ] + psi_random[ 0 ];
    }
    
    double Winv = 1/W[i];
    like += -0.50*log( 2*M_PI*Winv ) - 1/( 2*Winv )*pow( H[i] - psi_val, 2 );
  }
  
  // Return output
  return like;
}




// Function :: Calculate beta-binomial log-density (individual)
double log_beta_binomial_cpp( double indicate, double a, double b ){
  
  double post_a = indicate + a;
  double post_b = 1 - indicate + b;
  double log_indicator = lgamma( post_a ) + lgamma( post_b ) - lgamma( post_a + post_b ) - ( lgamma( a ) + lgamma( b ) - lgamma( a + b ) );
  
  // Return output
  return log_indicator ;
}

// Function :: Calculate normal log-density ( univariate )
double log_normal_cpp( double value, double mu, double sigma2 ){
  double log_normal = -0.50*log( 2*M_PI*sigma2 ) - 1/( 2*sigma2 )*pow( value - mu ,2 );
  
  // Return output
  return log_normal; 
}

// Function :: Calculate folded normal log-density ( univariate )
double log_fold_normal_cpp( double value, double mu, double sigma2 ){
  double log_fold_normal = log( pow( 2*M_PI*sigma2, -0.50 )*exp( - 1/( 2*sigma2 )*pow( value - mu ,2 )) + pow( 2*M_PI*sigma2, -0.50 )*exp( - 1/( 2*sigma2 )*pow( value + mu ,2 ) ) );
  
  // Return output
  return log_fold_normal; 
}

// Function :: Rescale beta_temp and xi_temp
List rescaled( arma::vec beta_temp, arma::vec xi_temp, arma::vec Ustar_dems){
  int P = Ustar_dems.size() - 1;
  
  for( int p = 0; p < P; ++p ){
    int Rp = Ustar_dems[ p + 1 ] - Ustar_dems[ p ]; 
    double summed = 0;
    
    for( int r = Ustar_dems[ p ]; r < Ustar_dems[ p + 1 ]; ++r){
      summed += std::abs( xi_temp[ r ] );
    }
    
    for( int r = Ustar_dems[ p ]; r < Ustar_dems[ p + 1 ]; ++r){
      xi_temp[ r ] = ( Rp/summed )*xi_temp[ r ];
    }
    
    beta_temp[ 3*p ] = ( summed/Rp )*beta_temp[ 3*p ];
  }
  
  List rescaled( 2 );
  rescaled[ 0 ] = beta_temp;
  rescaled[ 1 ] = xi_temp;
  return rescaled;
}

// Updates

// Update auxillary parameters W 
arma::vec update_W( arma::mat Ustar, arma::vec xi, arma::mat Xbar, arma::vec Ustar_dems, arma::vec beta_temp, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject ){
  int obs = subject.size();
  int S = beta_temp.size();
  int D = K_temp.size();
  
  // Make a home for W updates
  arma::vec updated_W( obs ); 
  updated_W.zeros();
  
  // Make sX
  arma::mat sX( obs, S); 
  sX.zeros();
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make K matrix
  arma::mat K_mat( D, D );
  K_mat.zeros();
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Update each W individually
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    arma::mat phi_j(1,1);
    phi_j = sX.row(j)*beta_temp + Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() ;
    updated_W[ j ] = help::samplepg( phi_j[ 0 ] );
  }
  
  return updated_W;
  
}

// Jointly updates beta and v Between step
List between_step_beta( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::vec v_temp, arma::vec t2_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, IntegerVector fixed_avail, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, double a, double b ){
  int S = beta_temp.size();
  
  // Randomly select a selectable fixed effect 
  int s = help::sample_cpp( fixed_avail );
 
  // Set proposals to current 
  arma::vec beta_proposal( S );
  arma::vec v_proposal( S );
  beta_proposal = beta_temp;
  v_proposal = v_temp;
  
  // Add
  if( v_temp[ s ] == 0 ){
    
    // Update proposal ( Normal(current_beta, 1)  )
    double beta_prop = beta_temp[ s ] + sqrt( 1 )*rnorm( 1 )[ 0 ];
    beta_proposal[ s ] = beta_prop;
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_proposal, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + help::log_normal_cpp( beta_prop, 0, t2_temp[ s ] ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      beta_temp[ s ] = beta_proposal[ s ];
      v_temp[ s ] = 1;
      
      
    }
    
  }else{
    
    // Delete  
    
    // Update proposal ( Normal(current_beta,1) )
    double beta_prop = 0;
    beta_proposal[ s ] = beta_prop;
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_proposal, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + help::log_normal_cpp( beta_temp[ s ], 0, t2_temp[ s ] ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      beta_temp[ s ] = 0;
      v_temp[ s ] = 0;
    }
  }
  
  List between_beta( 2 );
  between_beta[ 0 ] = beta_temp;
  between_beta[ 1 ] = v_temp;
  return between_beta;
}


// Function :: Update beta Within  
arma::vec within_beta( arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::vec t2_temp, arma::vec v_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat beta_update( 1, S );
  beta_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX and reduce based on v_temp and T2_v matrix 
  arma::mat sX( obs, S); 
  sX.zeros();
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  int v_dim = 0;
  v_dim = sum( v_temp ); 
  arma::mat sX_v(obs, v_dim);
  arma::mat T2_v( v_dim, v_dim );
  sX_v.zeros();
  T2_v.zeros();
  
  int count_v = 0; 
  for( int s = 0; s < S; ++s){
    if( v_temp[ s ] == 1 ){
      sX_v.col( count_v ) = sX.col( s );
      T2_v( count_v, count_v) = 1/t2_temp[ s ];
      count_v += 1; 
    }
  }
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_beta( v_dim, v_dim );
  arma::mat mu_beta( v_dim, 1 );
  arma::mat mu_beta_inside( v_dim, 1 );
  V_beta.zeros();
  mu_beta.zeros();
  mu_beta_inside.zeros();
  
  // Update for each individuals zeta_temp
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    
    V_beta += W[ j ]*sX_v.row( j ).t()*sX_v.row( j );
    mu_beta_inside += W[ j ]*sX_v.row( j ).t()*( H[ j ] -  Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
    
  }
  
  V_beta += T2_v; 
  V_beta = inv( V_beta );
  mu_beta = mu_beta_inside;
  
  mu_beta = V_beta*mu_beta;
  
  arma::mat beta_v_update( 1, v_dim );
  beta_v_update.zeros();
  
  beta_v_update = help::mvrnormArma( 1, mu_beta, V_beta ); 
  
  // Re-index the beta updates appropriately  
  count_v = 0; 
  for( int s = 0; s < S; ++s){
    if( v_temp[ s ] == 1 ){
      beta_update[ s ] = beta_v_update[ count_v ];
      count_v += 1; 
    }
  }
  
  return beta_update.t();
}  

// Function :: Sample new cluster for beta
double new_beta_cluster( arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::vec t2_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp, int S_index ){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat beta_update( 1, 1 );
  beta_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX  
  arma::mat sX( obs, S); 
  sX.zeros();
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  arma::vec sX_s = sX.col( S_index );
  double t2 = t2_temp[ S_index ];
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_beta( 1, 1 );
  arma::mat mu_beta( 1, 1 );
  arma::mat mu_beta_inside( 1, 1 );
  V_beta.zeros();
  mu_beta.zeros();
  mu_beta_inside.zeros();
  
  // Update for each individuals zeta_temp
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    
    V_beta += W[ j ]*sX_s.row( j )*sX_s.row( j );
    double other_beta = 0;
    for( int s = 0; s < S; ++s){
      if( s != S_index ){
        other_beta += sX( j, s )*beta_temp[ s ];
      }
    }
    mu_beta_inside += W[ j ]*sX_s.row( j )*( H[ j ] - other_beta - Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
  }
  
  V_beta += t2; 
  V_beta = 1/V_beta ;
  mu_beta = mu_beta_inside;
  
  mu_beta = V_beta*mu_beta;
  
  beta_update = help::mvrnormArma( 1, mu_beta, V_beta ); 
  
  return beta_update[ 0 ];
} 

// Update cluster assignment for each of the betas 
List cluster_beta_cpp( arma::vec cluster_temp, arma::vec cluster_count_temp, arma::vec cluster_beta_temp, double vartheta, arma::vec beta_temp, arma::vec v_temp,  arma::vec Y, arma::vec W, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, arma::vec t2_temp, arma::vec subject_dems, arma::vec mu_temp  ){
  int S = beta_temp.size();
  int obs = Y.size();
  int D = K_temp.size();
  
  // Update cluster assignment for each beta 
  for( int s = 0; s < S; ++s ){
    // Only if included
   
    if( ( v_temp[ s ] == 1 ) & ( s % 3 != 0 ) ){            
    
      // Get marignal likelihood of data
      arma::vec H = (Y - 0.5)/W;          
      arma::vec beta_marginal = beta_temp;           
      beta_marginal[ s ] = 0;              
   
      // Make sX
      arma::mat sX( obs, S);          
      sX.zeros();          
      sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );          
      
      // Make K matrix
      arma::mat K_mat( D, D );        
      K_mat.zeros();          
      for( int i = 0; i < D; ++i ){          
        K_mat( i, i ) = K_temp[ i ];           
      }
      
      // Sum over the observations to calculate the marginal likelihood 
      double scale = 0;            
      double C = 0;            
      double numer = 0;             
      double denom = 0;            
      
      for( int j = 0; j < obs; ++j ){
        int sub = subject[ j ];            
        arma::mat phi_j( 1, 1 );            
        phi_j = sX.row(j)*beta_marginal + Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() ;            
        
        scale += -0.5*log( 2*M_PI/W[ j ] );            
        C += ( pow( H[ j ], 2 ) - 2*H[ j ]*phi_j[ 0 ] + pow( phi_j[ 0 ], 2) )*W[ j ];             
        numer += ( H[ j ]*sX( j, s ) + sX( j, s )*phi_j[ 0 ] )*W[ j ] ;            
        denom += pow( sX( j, s ), 2)*W[ j ];            
      }
      
      double marginal = scale - 0.5*log( 2*M_PI*t2_temp[ s ] ) - 0.5*C + pow( numer, 2 )/( 2*( denom + t2_temp[ s ] ) ) + 0.5*log( 2*M_PI ) - 0.5*log( denom + t2_temp[ s ] ) ;            
      double last_prob = log( vartheta ) + marginal;             
      
      // Get cluster assignment 
      int cluster_s = cluster_temp[ s ];              
      
      // Subtract 1 from cluster
      cluster_count_temp[ cluster_s ] += -1;            
      
      // If cluster_s is now empty, remove the potential beta
      if( cluster_count_temp[ cluster_s ] == 0 ){            
        cluster_beta_temp[ cluster_s ] = 0;            
      }
      
      // Build vector of available cluster labels, count non-zero clusters, make vector of non-zero betas, make vector of non-zero beta indicies
      int non_zero_clusters = 0;             
      NumericVector non_zero_betas( 0 );            
      IntegerVector beta_index( 0 );            
      IntegerVector cluster_avail( 0 );            
      
      for( int j = 0; j < S; ++j ){
        // Available clusters 
        if( ( cluster_beta_temp[ j ] == 0 ) & ( j % 3 != 0 )  ){           // Adjust so that only the beta_bar that are zero are available
          cluster_avail.push_back( j );
        }
        // # of non-zero clusters, non-zero betas, non-zero beta indicies
        if( cluster_beta_temp[ j ] != 0 ){            
          non_zero_betas.push_back( cluster_beta_temp[ j ] );            
          beta_index.push_back( j );            
          non_zero_clusters += 1;            
        }
      }
      
      // Make vector for probabilities to live 
      arma::vec prob( non_zero_clusters + 1 );            
      prob.zeros();            
      
      // Make likelihood for the clusters           
      arma::vec likelihood = help::like_pg_beta_clust( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject, non_zero_betas, s, non_zero_clusters );
      
      
      // Append to probabilities for existing clusters
      for( int l = 0; l < non_zero_clusters; ++l ){           
        prob[ l ] = likelihood[ l ] + log( cluster_count_temp[ beta_index[ l ] ] );
      }
      
      // Append probability for new cluster
      prob[ non_zero_clusters ] = last_prob;           
      
      // Put on zero to one scale 
      // Use log sum exp trick to prevent numerical underflow
      double maxi = max( prob );           
      double thresh = log(.00000000000000000001) - log( prob.size() );           
      double sum_denom = 0;           
      
      for( int k = 0; k < ( non_zero_clusters + 1 ); ++k ){           
        if ( prob[ k ] - maxi > thresh ){           
          sum_denom += exp( prob[ k ] - maxi );           
          prob[ k ] = exp( prob[ k ] - maxi );           
        }else{           
          prob[ k ] = 0;           
        }
      }
      // Normalize probability vector 
      prob = prob/sum_denom;           
      
      // Sample from multinomial
      IntegerVector sampled_cluster( non_zero_clusters + 1 );           
      R::rmultinom(1, prob.begin(), non_zero_clusters + 1, sampled_cluster.begin());           
      int sampled = 0;           
      
      for( int m = 0; m < ( non_zero_clusters + 1 ); ++m ){           
        if( sampled_cluster[ m ] == 1 ){           
          sampled = m;           
        }           
      }           
      
      // Adjust according to the sampled cluster 
      if( sampled != non_zero_clusters ){           
        int get = beta_index[ sampled ];           
        beta_temp[ s ] = cluster_beta_temp[ get ];           // Should be okay if the cluster beta temp for beta* are zero
        cluster_temp[ s ] = get;           
        cluster_count_temp[ get ] += 1;           
      }else{
        // Sample a new beta_c
        double samp_beta = help::new_beta_cluster( Y, W, subject, beta_temp, t2_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp, s );
        beta_temp[ s ] = samp_beta;           
        int get = cluster_avail[ 0 ];           
        cluster_temp[ s ] = get;           
        cluster_beta_temp[ get ] = samp_beta;           
        cluster_count_temp[ get ] += 1;           
      }
      
    }
  }
  
  // Return beta cluster details 
  List return_cluster_beta( 4 );
  return_cluster_beta[ 0 ] = cluster_temp;
  return_cluster_beta[ 1 ] = cluster_count_temp;
  return_cluster_beta[ 2 ] = cluster_beta_temp;
  return_cluster_beta[ 3 ] = beta_temp;
  
  return return_cluster_beta ;
}

// Jointly updates beta and v Between step for DP
List between_step_beta_DP( arma::vec cluster_temp, arma::vec cluster_count_temp, arma::vec cluster_beta_temp, double vartheta, arma::vec Y, arma::vec W, arma::vec beta_temp, arma::vec v_temp, arma::vec t2_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, IntegerVector fixed_avail, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, double a, double b ){
  int S = beta_temp.size();
  
  // Randomly select a selectable fixed effect 
  int s = help::sample_cpp( fixed_avail );
 
  // Set proposals to current 
  arma::vec beta_proposal( S );
  arma::vec v_proposal( S );
  beta_proposal = beta_temp;
  v_proposal = v_temp;
  
  // Add
  if( v_temp[ s ] == 0 ){
    
    // Update proposal ( Normal(current_beta, 1)  )
    double beta_prop = beta_temp[ s ] + sqrt( t2_temp[ s ] )*rnorm( 1 )[ 0 ];
    beta_proposal[ s ] = beta_prop;
    
    // Calculate ratio for DP and Normal 
    double beta_new = 0;
    if( s % 3 != 0 ){
      beta_new = log( vartheta ) - log( vartheta + sum( v_temp ) - 1 ) + help::log_normal_cpp( beta_prop, 0, t2_temp[ s ] );
    }else{
      beta_new = help::log_normal_cpp( beta_prop, 0, t2_temp[ s ] );
    }
    
    double r = ( help::log_like_pg( Y, W, beta_proposal, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + beta_new ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      // Adjust selection information 
      beta_temp[ s ] = beta_proposal[ s ];
      v_temp[ s ] = 1;
      
      // Adjust cluster details if DP update ( Note, this depends on the structure of the beta and thus requires more attention to input pre-processing via the wrapper function )
      if( s % 3 != 0 ){
        // Build vector of available cluster labels
        IntegerVector cluster_avail( 0 );
        for( int j = 0; j < S; ++j ){
          // Available clusters 
          if( cluster_beta_temp[ j ] == 0 ){
            cluster_avail.push_back( j );
          }
        }
        
        // Adjust cluster information 
        int get = cluster_avail[ 0 ];
        cluster_temp[ s ] = get;
        cluster_beta_temp[ get ] = beta_proposal[ s ];
        cluster_count_temp[ get ] += 1;
      }
      
    } // End accept 
    
  }else{
    
    // Delete  
    
    // Update proposal ( Normal(current_beta,1) )
    double beta_prop = 0;
    beta_proposal[ s ] = beta_prop;
    
    // Get cluster information 
    int cluster_s = cluster_temp[ s ];
    
    // Adjust prior for DP or not
    double beta_old = 0;
    if( s % 3 != 0 ){
      // Set prior for beta based on size of cluster 
      if( cluster_count_temp[ cluster_s ] == 1 ){
        beta_old = log( vartheta ) - log( vartheta + sum( v_temp ) - 1 ) +  help::log_normal_cpp( beta_prop, 0, t2_temp[ s ] );
      }else{
        beta_old = log( cluster_count_temp[ cluster_s ] - 1 ) - log( vartheta + sum( v_temp ) - 1 );
      }
    }else{
      beta_old = help::log_normal_cpp( beta_prop, 0, t2_temp[ s ] );
    }
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_proposal, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + beta_old );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      // Adjust selection information 
      beta_temp[ s ] = 0;
      v_temp[ s ] = 0;
      
      if( s % 3 != 0 ){ 
        // Adjust cluster information 
        // Subtract 1 from cluster
        cluster_count_temp[ cluster_s ] += -1; // Change dimension of current cluster
        if( cluster_count_temp[ cluster_s ] == 0 ){
          cluster_beta_temp[ cluster_s ] = 0; 
        }
        cluster_temp[ s ] = -1;                // Set current cluster to -1 since it is not included
      }
    }
  }
  
  List between_beta( 5 );
  between_beta[ 0 ] = beta_temp;
  between_beta[ 1 ] = v_temp;
  between_beta[ 2 ] = cluster_temp; 
  between_beta[ 3 ] = cluster_count_temp;
  between_beta[ 4 ] = cluster_beta_temp;
  
  return between_beta;
}

// Function :: Update beta Within for DP
List within_beta_DP( arma::vec cluster_temp, arma::vec cluster_count_temp, arma::vec cluster_beta_temp, double vartheta, arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::vec t2_temp, arma::vec v_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat beta_update( 1, S ); 
  beta_update.zeros();  
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){     
    K_mat( i, i ) = K_temp[ i ];     
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;       
  
  // Make sX and sum over cluster assignment before reducing 
  arma::mat sX( obs, S);     
  sX.zeros();     
  sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );      
  
  arma::mat sX_clusters( obs, S );            
  sX_clusters.zeros();     
  
  int non_zero_clusters = 0;       
  IntegerVector beta_index( 0 );      
  
  int non_zero_non_linear = 0;
  IntegerVector non_linear_index( 0 );
  
  for( int s = 0; s < S; ++s ){
    
    if( cluster_beta_temp[ s ] != 0 ){       
      beta_index.push_back( s );       
      non_zero_clusters += 1;       
    }
    
    if( v_temp[ s ] == 1 ){      
      
      if( s % 3 != 0 ){
        // Get cluster assignment 
        int cluster_s = cluster_temp[ s ];       
        
        // Add column to cluster sum 
        sX_clusters.col( cluster_s ) += sX.col( s );
        
      }else{
        // Adjust index and number of included non-linear terms 
        non_linear_index.push_back( s ); 
        non_zero_non_linear += 1;
      }
    }
  }
  
  // Combine all the sX for the non-zero terms in the model 
  double beta_size = non_zero_clusters + non_zero_non_linear;
  arma::mat sX_sum( obs, beta_size );
  sX_sum.zeros();
  
  // Get non-zero columns in sX_clusters 
  IntegerVector linear( 0 );
  for( int l = 0; l < non_zero_clusters; ++l ){
    linear.push_back( l );
  }
  
  sX_sum.cols( as<uvec>( linear ) ) = sX_clusters.cols(  as<uvec>( beta_index ) );
  
  // Add the non-zero non-linear interation terms 
  IntegerVector non_linear( 0 );
  for( int n = non_zero_clusters; n < beta_size; ++n ){
    non_linear.push_back( n );
  }
  sX_sum.cols( as<uvec>( non_linear ) ) = sX.cols(  as<uvec>( non_linear_index ) );
  
  
  // Set T ( Assumes all t_2 are the same )
  arma::mat T( beta_size, beta_size );
  T.zeros();
  for( int t = 0; t < beta_size; ++t ){
    T( t, t ) = 1/t2_temp[ 0 ];
  }
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_beta_c( beta_size, beta_size );
  arma::mat mu_beta_c( beta_size, 1 );
  arma::mat mu_beta_inside_c( beta_size, 1 );
  V_beta_c.zeros();
  mu_beta_c.zeros();
  mu_beta_inside_c.zeros();
  
  
  // Update for each individuals zeta_temp
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    V_beta_c += W[ j ]*sX_sum.row( j ).t()*sX_sum.row( j );
    mu_beta_inside_c += W[ j ]*sX_sum.row( j ).t()*( H[ j ] -  Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
    
  }
  
  V_beta_c += T; 
  
  V_beta_c = inv( V_beta_c );
  
  mu_beta_c = mu_beta_inside_c;
  
  mu_beta_c = V_beta_c*mu_beta_c;
  
  arma::mat beta_c_update( 1, beta_size  );
  beta_c_update.zeros();
  
  beta_c_update = help::mvrnormArma( 1, mu_beta_c, V_beta_c ); 
  
  // Re-index the beta updates appropriately and return temp_cluster_beta 
  for( int b = 0; b < non_zero_clusters; ++b ){
    for( int s = 0; s < S; ++ s){
      if( beta_index[ b ] == cluster_temp[ s ] ){
        beta_update[ s ] = beta_c_update[ b ];
      }
      cluster_beta_temp[ beta_index[ b ] ] = beta_c_update[ b ];
    }
  }
  
  // Update the non-liner components 
  for( int n = 0; n < non_zero_non_linear; ++n ){
    beta_update[ non_linear_index[ n ] ] = beta_c_update[ ( non_zero_clusters + n ) ];
  }
  
  List within_cluster_beta( 2 );
  within_cluster_beta[ 0 ] = beta_update.t();
  within_cluster_beta[ 1 ] = cluster_beta_temp;
  
  return within_cluster_beta;
  
} 

// Function :: Update concentration parameter vartheta
double update_vartheta( double vartheta_temp, arma::vec v_temp, arma::vec cluster_beta_temp, double a_vartheta, double b_vartheta ){
  int S = cluster_beta_temp.size();
  double M = 0 ;
  double n_v = sum( v_temp ) ;
  
  // Get the number of clusters
  for( int s = 0; s < S; ++s ){
    if( cluster_beta_temp[ s ] != 0 ){
      M += 1;
    }
  }
  
  // Update nuisance parameter eta_vartheta 
  double eta_vartheta = R::rbeta( vartheta_temp + 1, n_v );
  
  // Update vartheta
  double b = b_vartheta - log( eta_vartheta ) ;
  double prob = ( a_vartheta + M - 1)/( n_v*b + a_vartheta + M - 1 );
  double pi_vartheta = rbinom( 1, 1, prob )[0];
  
  if( pi_vartheta == 1 ){
    vartheta_temp = rgamma(1, ( a_vartheta + M ) , 1/b )[0];
  }else{
    vartheta_temp = rgamma(1, ( a_vartheta + M - 1 ), 1/b )[0];
  }
  
  return vartheta_temp;
}


// Function :: Update xi 
arma::vec update_xi( arma::vec Y, arma::vec W, arma::vec subject, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject_dems, arma::vec mu_temp){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int P = Ustar_dems.size() - 1;
  int Rp = Ustar_dems[ Ustar_dems.size() - 1 ];
  int D = K_temp.size();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::vec H( obs );
  H.zeros();
  
  arma::mat xi_update( 1, Rp );
  xi_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make Identity
  arma::mat I( Rp, Rp );
  I.zeros();
  for( int r = 0; r < Rp; ++r){
    I( r, r ) = 1;
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make X*
  arma::mat Xstar( obs, Rp );
  Xstar.zeros();
  Xstar = help::make_Xstar( Ustar, beta_temp, Xbar, Ustar_dems );
  
  // Make V_xi and mu_xi (and inside)
  arma::mat V_xi( Rp, Rp );
  arma::mat mu_xi( Rp, 1 );
  arma::mat mu_xi_inside( Rp, 1 );
  V_xi.zeros();
  mu_xi.zeros();
  mu_xi_inside.zeros();
  
  // Make beta_bar
  arma::vec beta_bar( S - P );
  for( int p = 0; p < P; ++p ){
    beta_bar[ 2*p ] = beta_temp[ 3*p + 1 ];
    beta_bar[ 2*p + 1 ] = beta_temp[ 3*p + 2 ] ;
  }
  
  // Update for each individuals xi
  for( int j = 0; j < obs; ++j ){
    int sub = subject[ j ];
    V_xi += W[ j ]*Xstar.row( j ).t()*Xstar.row( j );
    mu_xi_inside += W[ j ]*Xstar.row( j ).t()*( H[ j ] -  Xbar.row(j)*beta_bar - Z.row( j )*K_mat*Gamma_temp*zeta_temp.row( sub ).t() );
  }
  
  V_xi += I; 
  V_xi = inv( V_xi );
  mu_xi = mu_xi_inside + mu_temp;
  mu_xi = V_xi*mu_xi;
  xi_update = help::mvrnormArma( 1, mu_xi, V_xi ); 
  
  return xi_update.t();
}

// Function :: Update mu_rp 
arma::vec update_mu_rp( arma::vec xi_temp ){
  int Rp = xi_temp.size();
  arma::vec xi_update( Rp );
  
  for( int rp = 0 ; rp < Rp; ++rp){
    
    // Get probability based on xi_temp 
    double prob = 1/(1 + exp( -2*xi_temp[ rp ] ) );
    
    // +- 1 based on sampled mu_ind
    int mu_ind = rbinom( 1, 1, prob )[0];
    
    if( mu_ind == 1){
      xi_update[ rp ] = 1;
    }else{
      xi_update[ rp ] = -1;
    }
    
  }
  return xi_update;
}

// Function :: Update t2
arma::vec update_t2_s( arma::vec beta_temp, double a_0, double b_0 ){
  int S = beta_temp.size();
  
  arma::vec t2_s_update( S );
  t2_s_update.zeros();
  
  for( int s = 0; s < S; ++s ){
    double a_post = a_0 + 1/2;
    double b_post = b_0 + pow( beta_temp[ s ], 2 )/2;
    t2_s_update( s ) = 1/rgamma(1, a_post, 1/b_post )[0];
  }
  return t2_s_update;
}


// Jointly updates K and lambda Between step
List between_step_K ( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::vec lambda_temp, IntegerVector random_avail, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, double a, double b, double m_star, double v_star, double m_0, double v_0 ){
  int D = K_temp.size();
  
  // Randomly select a selectable random effect 
  int d = help::sample_cpp( random_avail );
  
  // Set proposals to current 
  arma::vec K_proposal( D );
  arma::vec lambda_proposal( D );
  K_proposal = K_temp;
  lambda_proposal = lambda_temp;
  
  // Add
  if( lambda_temp[ d ] == 0 ){
    
    // Update proposal ( Folded Normal(m*,v*) )
    double K_prop = m_star + sqrt( v_star )*rnorm( 1 )[ 0 ];
    K_proposal[ d ] = std::abs( K_prop );
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_proposal, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + help::log_fold_normal_cpp( K_prop, m_0, v_0 ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) + help::log_fold_normal_cpp( K_prop, m_star, v_star ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      K_temp[ d ] = K_proposal[ d ];
      lambda_temp[ d ] = 1;
    }
    
  }else{
    
    // Delete  
    
    // Update proposal ( Folded Normal(m*,v*) )
    double K_prop = 0;
    K_proposal[ d ] = K_prop;
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_proposal, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) + help::log_fold_normal_cpp( K_temp[ d ], m_star, v_star ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + help::log_fold_normal_cpp( K_temp[ d ], m_0, v_0 ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      K_temp[ d ] = 0;
      lambda_temp[ d ] = 0;
    }
  }
  
  List between_K( 2 );
  between_K[ 0 ] = K_temp;
  between_K[ 1 ] = lambda_temp;
  return between_K;
}

// Updates selected K and lambda Within step
arma::vec within_step_K( arma::vec Y, arma::vec subject, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::vec lambda_temp, arma::mat Gamma_temp, arma::mat zeta_temp, double m_0, double v_0 ){
  int D = K_temp.size();
  int obs = W.size();
  int S = beta_temp.size();
  arma::vec H( obs );
  H.zeros();
  arma::mat sX( obs, S );
  sX.zeros();
  
  // Make home for K updates
  arma::vec K_update( D );
  K_update.zeros();
  
  // Get Z*
  arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  for( int d = 0; d < D; ++d ){
    if( lambda_temp[ d ] == 1 ){
      double var_kd = 0;
      arma::mat mean_kd( 1, 1 );
      mean_kd.zeros();
      
      // Iterate over the subjects 
      for( int j = 0; j < obs; ++j ){
        mean_kd += W[ j ]*Zstar( j, d )*( H[ j ] - sX.row( j )*beta_temp - Zstar.row( j )*K_temp + Zstar( j, d )*K_temp[ d ] );
        var_kd += W[ j ]*pow( Zstar( j, d ), 2 ); 
      }
      
      var_kd += 1/v_0;
      var_kd = 1/var_kd;
      mean_kd += m_0/v_0;
      mean_kd = var_kd*mean_kd[ 0 ]; 
      
      K_update[ d ] = std::abs( mean_kd[ 0 ] + sqrt( var_kd )*rnorm( 1 )[ 0 ] ) ;
    }
  }
  
  return K_update;
}


// Function :: Sample new cluster for kappa
double new_kappa_cluster( arma::vec Y, arma::vec subject, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::vec lambda_temp, arma::mat Gamma_temp, arma::mat zeta_temp, double m_star, double v_star, int D_index  ){
  int D = K_temp.size();
  int obs = W.size();
  int S = beta_temp.size();
  arma::vec H( obs );
  H.zeros();
  arma::mat sX( obs, S );
  sX.zeros();
  
  // Get Z*
  arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
  
  // Get dth column of Z*
  arma::vec Zstar_d = Zstar.col( D_index );
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Set up for FCD proposal 
  double var_k = 0;
  arma::mat mean_k( 1, 1 );
  mean_k.zeros();
  
  // Update for each individuals contribution
  for( int j = 0; j < obs; ++j ){
    
    var_k += W[ j ]*pow( Zstar_d[ j ], 2 ); 
    
    double other_kappa = 0;
    for( int d = 0; d < D; ++d ){
      if( d != D_index ){
        other_kappa += Zstar( j , d )*K_temp[ d ];
      }
    }
    
    mean_k += W[ j ]*Zstar_d.row( j )*( H[ j ] - sX.row( j )*beta_temp - Zstar.row( j )*K_temp + Zstar( j, D_index )*K_temp[ D_index ] );
    
  }
  
  var_k += 1/v_star;
  var_k = 1/var_k;
  mean_k += m_star/v_star;
  mean_k = var_k*mean_k[ 0 ]; 
  
  double K_update = std::abs( mean_k[ 0 ] + sqrt( var_k )*rnorm( 1 )[ 0 ] ) ;
  
  return K_update;
}

// Update cluster assignment for each of the kappas 
List cluster_kappa_cpp( arma::vec cluster_K_temp, arma::vec cluster_count_K_temp, arma::vec cluster_kappa_temp, double sA, arma::vec beta_temp, arma::vec v_temp,  arma::vec Y, arma::vec W, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::vec lambda_temp, arma::mat zeta_temp, arma::vec subject, arma::vec t2_temp, arma::vec subject_dems, arma::vec mu_temp, double m_star, double v_star, double m_0, double v_0  ){
  
  int S = beta_temp.size();
  int obs = Y.size();
  int D = K_temp.size();
  
  // Update cluster assignment for each kappa 
  for( int d = 0; d < D; ++d ){
    // Only if included
    if( lambda_temp[ d ] == 1 ){
      
      // Get marignal likelihood of data
      arma::vec H = (Y - 0.5)/W;
      arma::vec kappa_marginal = K_temp;
      kappa_marginal[ d ] = 0; 
      
      // Make sX
      arma::mat sX( obs, S);
      sX.zeros();
      sX = help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
      
      // Get Z*
      arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
      
      // Sum over the observations to calculate the marginal likelihood 
      double scale = 0;
      double C = 0;
      double numer = 0; 
      double denom = 0;
      
      for( int j = 0; j < obs; ++j ){
        arma::mat phi_j( 1, 1 );
        phi_j = sX.row(j)*beta_temp + Zstar.row( j )*kappa_marginal ;
        
        scale += -0.5*log( 2*M_PI/W[ j ] );
        C += ( pow( H[ j ], 2 ) - 2*H[ j ]*phi_j[ 0 ] + pow( phi_j[ 0 ], 2) )*W[ j ]; 
        numer += ( H[ j ]*Zstar( j, d ) + Zstar( j, d )*phi_j[ 0 ] )*W[ j ] ;
        denom += pow( Zstar( j, d ), 2)*W[ j ];
      }
      
      C += pow( m_0, 2)/v_0;
      double numer_pos = numer + m_0/v_0;
      double numer_neg = numer - m_0/v_0;
      arma::vec marginal( 2 );
      marginal[ 0 ] = scale - 0.5*log( 2*M_PI*v_0 ) - 0.5*C  +  pow( numer_pos, 2 )/( 2*( denom + 1/v_0 ) ) + 0.5*log( 2*M_PI ) - 0.5*log( denom + 1/v_0 ) ;
      marginal[ 1 ] = scale - 0.5*log( 2*M_PI*v_0 ) - 0.5*C  +  pow( numer_neg, 2 )/( 2*( denom + 1/v_0 ) ) + 0.5*log( 2*M_PI ) - 0.5*log( denom + 1/v_0 ) ;
      
      double max_m = max( marginal );
      
      double log_marg = max_m + log( exp( marginal[ 0 ] - max_m ) + exp( marginal[ 1 ] - max_m ) );
      double last_prob = log( sA ) + log_marg; 
      
      // Get cluster assignment 
      int cluster_d = cluster_K_temp[ d ];
      
      // Subtract 1 from cluster
      cluster_count_K_temp[ cluster_d ] += -1;
      
      // If cluster_s is now empty, remove the potential kappa
      if( cluster_count_K_temp[ cluster_d ] == 0 ){
        cluster_kappa_temp[ cluster_d ] = 0;
      }
      
      // Build vector of available cluster labels, count non-zero clusters, make vector of non-zero kappas, make vector of non-zero kappa indicies
      int non_zero_clusters_K = 0; 
      NumericVector non_zero_kappas( 0 );
      IntegerVector kappa_index( 0 );
      IntegerVector cluster_K_avail( 0 );
      
      for( int w = 0; w < D; ++w ){
        // Available clusters 
        if( cluster_kappa_temp[ w ] == 0 ){
          cluster_K_avail.push_back( w );
        }
        // # of non-zero clusters, non-zero betas, non-zero beta indicies
        if( cluster_kappa_temp[ w ] != 0 ){
          non_zero_kappas.push_back( cluster_kappa_temp[ w ] );
          kappa_index.push_back( w );
          non_zero_clusters_K += 1;
        }
      }
      
      // Make vector for probabilities to live 
      arma::vec prob( non_zero_clusters_K + 1 );
      prob.zeros();
      
      // Make likelihood for the clusters
      arma::vec likelihood = help::like_pg_kappa_clust( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject, non_zero_kappas, d, non_zero_clusters_K );
      
      
      // Append to probabilities for existing clusters
      for( int l = 0; l < non_zero_clusters_K; ++l ){
        prob[ l ] = likelihood[ l ] + log( cluster_count_K_temp[ kappa_index[ l ] ] );
      }
      
      // Append probability for new cluster 
      prob[ non_zero_clusters_K ] = last_prob;
      
      // Put on zero to one scale 
      // Use log sum exp trick to prevent numerical underflow
      double maxi = max( prob );
      double thresh = log(.00000000000000000001) - log( prob.size() );
      double sum_denom = 0;
      
      for( int k = 0; k < ( non_zero_clusters_K + 1 ); ++k ){
        if ( prob[ k ] - maxi > thresh ){
          sum_denom += exp( prob[ k ] - maxi );
          prob[ k ] = exp( prob[ k ] - maxi );
        }else{
          prob[ k ] = 0;
        }
      }
      // Normalize probability vector 
      prob = prob/sum_denom;
      
      // Sample from multinomial
      IntegerVector sampled_cluster( non_zero_clusters_K + 1 );
      R::rmultinom(1, prob.begin(), non_zero_clusters_K + 1, sampled_cluster.begin() );
      int sampled = 0;
      
      for( int m = 0; m < ( non_zero_clusters_K + 1 ); ++m ){
        if( sampled_cluster[ m ] == 1 ){
          sampled = m;
        }
      }
      
      // Adjust according to the sampled cluster 
      if( sampled != non_zero_clusters_K ){
        int get = kappa_index[ sampled ];
        K_temp[ d ] = cluster_kappa_temp[ get ];
        cluster_K_temp[ d ] = get;
        cluster_count_K_temp[ get ] += 1;
      }else{
        // Sample a new kappa_d
        
        double samp_kappa = help::new_kappa_cluster( Y, subject, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, lambda_temp, Gamma_temp, zeta_temp, m_star, v_star, d );
        K_temp[ d ] = samp_kappa;
        int get = cluster_K_avail[ 0 ];
        cluster_K_temp[ d ] = get;
        cluster_kappa_temp[ get ] = samp_kappa;
        cluster_count_K_temp[ get ] += 1;
      }
      
    }
  }
  
  List return_cluster_kappa( 4 );
  return_cluster_kappa[ 0 ] = cluster_K_temp;
  return_cluster_kappa[ 1 ] = cluster_count_K_temp;
  return_cluster_kappa[ 2 ] = cluster_kappa_temp;
  return_cluster_kappa[ 3 ] = K_temp;
  
  
  return return_cluster_kappa ;
}


// Jointly updates K and lambda Between step for DP 
List between_step_K_DP( arma::vec cluster_K_temp, arma::vec cluster_count_K_temp, arma::vec cluster_kappa_temp, double sA, arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::vec lambda_temp, IntegerVector random_avail, arma::mat Gamma_temp, arma::mat zeta_temp, arma::vec subject, double a, double b, double m_star, double v_star, double m_0, double v_0 ){
  int D = K_temp.size();
  
  // Randomly select a selectable random effect 
  int d = help::sample_cpp( random_avail );
  
  // Set proposals to current 
  arma::vec K_proposal( D );
  arma::vec lambda_proposal( D );
  K_proposal = K_temp;
  lambda_proposal = lambda_temp;
  
  // Add
  if( lambda_temp[ d ] == 0 ){
    
    // Update proposal ( Folded Normal(m*,v*) )
    double K_prop = m_star + sqrt( v_star )*rnorm( 1 )[ 0 ];
    K_proposal[ d ] = std::abs( K_prop );
    
    // Calculate ratio 
    double kappa_DP_new = log( sA ) - log( sA + sum( lambda_temp ) - 1);
    double r = ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_proposal, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + kappa_DP_new ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) + help::log_fold_normal_cpp( K_prop, m_star, v_star ) );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      // Adjust selection information 
      K_temp[ d ] = K_proposal[ d ];
      lambda_temp[ d ] = 1;
      
      // Build vector of available cluster labels 
      IntegerVector cluster_avail( 0 );
      for( int m = 0; m < D; ++m ){
        // Available clusters 
        if( cluster_kappa_temp[ m ] == 0 ){
          cluster_avail.push_back( m );
        }
      }
      
      // Adjust cluster information 
      int get = cluster_avail[ 0 ];
      cluster_K_temp[ d ] = get;
      cluster_kappa_temp[ get ] = K_proposal[ d ];
      cluster_count_K_temp[ get ] += 1;
      
    } // End accept 
    
  }else{
    
    // Delete  
    
    // Update proposal ( Folded Normal(m*,v*) )
    double K_prop = 0;
    K_proposal[ d ] = K_prop;
    
    // Get cluster information 
    int cluster_d = cluster_K_temp[ d ];
    
    // Set prior for beta based on size of cluster 
    double kappa_DP_old = 0;
    if( cluster_count_K_temp[ cluster_d ] == 1 ){
      kappa_DP_old = log( sA ) - log( sA + sum( lambda_temp ) - 1 ) +  help::log_fold_normal_cpp( K_prop, m_0, v_0 );
    }else{
      kappa_DP_old = log( cluster_count_K_temp[ cluster_d ] - 1 ) - log( sA + sum( lambda_temp ) - 1 );
    }
    
    // Calculate ratio 
    double r = ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_proposal, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 0, a, b ) + help::log_fold_normal_cpp( K_temp[ d ], m_star, v_star ) ) - ( help::log_like_pg( Y, W, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject ) + help::log_beta_binomial_cpp( 1, a, b ) + kappa_DP_old );
    
    // Calculate acceptance probability
    double a  = log( runif( 1 )[ 0 ] );
    
    // Determine acceptance
    if( a < r ){
      // Adjust selection information 
      K_temp[ d ] = 0;
      lambda_temp[ d ] = 0;
      
      // Adjust cluster information 
      cluster_count_K_temp[ cluster_d ] += -1;  
      if( cluster_count_K_temp[ cluster_d ] == 0 ){
        cluster_kappa_temp[ cluster_d ] = 0; 
      }
      cluster_K_temp[ d ] = -1;               
      
    }
  }
  
  
  
  List between_K( 5 );
  between_K[ 0 ] = K_temp;
  between_K[ 1 ] = lambda_temp;
  between_K[ 2 ] = cluster_K_temp; 
  between_K[ 3 ] = cluster_count_K_temp;
  between_K[ 4 ] = cluster_kappa_temp;
  
  return between_K;
}

// Updates selected K and lambda Within step
List within_step_K_DP( arma::vec cluster_K_temp, arma::vec cluster_count_K_temp, arma::vec cluster_kappa_temp, double sA, arma::vec Y, arma::vec subject, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::vec lambda_temp, arma::mat Gamma_temp, arma::mat zeta_temp, double m_0, double v_0 ){
  int D = K_temp.size();
  int obs = W.size();
  int S = beta_temp.size();
  arma::vec H( obs );
  H.zeros();
  arma::mat sX( obs, S );
  sX.zeros();
  
  // Get Z*
  arma::mat Zstar = help::make_Zstar( Z, zeta_temp, Gamma_temp, subject );
  
  // Get sum of each of the clusters 
  arma::mat Zstar_clusters( obs, D );
  Zstar_clusters.zeros();
  
  int non_zero_clusters = 0;
  IntegerVector kappa_index( 0 );
  
  for( int d = 0; d < D; ++d ){
    
    if( cluster_kappa_temp[ d ] != 0 ){
      kappa_index.push_back( d );
      non_zero_clusters += 1;
    }
    
    if( lambda_temp[ d ] == 1 ){
      
      // Get cluster assignment 
      int cluster_d = cluster_K_temp[ d ];
      
      // Add column to cluster sum 
      Zstar_clusters.col( cluster_d ) += Zstar.col( d );
    }
  }
  
  // Get non-zero columns in Zstar_clusters 
  arma::mat Zstar_sum( obs, non_zero_clusters );
  Zstar_sum.zeros();
  Zstar_sum = Zstar_clusters.cols(  as<uvec>( kappa_index ) );
  
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make home for K updates
  arma::vec K_update( non_zero_clusters );
  K_update.zeros();
  
  for( int d = 0; d < non_zero_clusters; ++d ){
    
    double var_kd = 0;
    arma::mat mean_kd( 1, 1 );
    mean_kd.zeros();
    
    // Iterate over the subjects 
    for( int j = 0; j < obs; ++j ){
      mean_kd += W[ j ]*Zstar_sum( j, d )*( H[ j ] - sX.row( j )*beta_temp - Zstar.row( j )*K_temp + Zstar_sum( j, d )*cluster_kappa_temp[ kappa_index[ d ] ] );
      var_kd += W[ j ]*pow( Zstar_sum( j, d ), 2 ); 
    }
    
    var_kd += 1/v_0;
    var_kd = 1/var_kd;
    mean_kd += m_0/v_0;
    mean_kd = var_kd*mean_kd[ 0 ]; 
    
    K_update[ d ] = std::abs( mean_kd[ 0 ] + sqrt( var_kd )*rnorm( 1 )[ 0 ] ) ;
  }
  
  // Re-index the kappa updates appropriately and return  cluster_kappa_temp
  for( int b = 0; b < non_zero_clusters; ++b ){
    for( int d = 0; d < D; ++ d){
      if( kappa_index[ b ] == cluster_K_temp[ d ] ){
        K_temp[ d ] = K_update[ b ];
      }
      cluster_kappa_temp[ kappa_index[ b ] ] = K_update[ b ];
    }
  }
  
  List within_cluster_kappa( 2 );
  within_cluster_kappa[ 0 ] = K_temp;
  within_cluster_kappa[ 1 ] = cluster_kappa_temp;
  
  return within_cluster_kappa;
  
}

// Function:: Update concentration parameter sA
double update_sA( double sA_temp, arma::vec lambda_temp, arma::vec cluster_kappa_temp, double a_sA, double b_sA ){
  int D = cluster_kappa_temp.size();
  double M_til = 0 ;
  double n_lambda = sum( lambda_temp ) ;
  
  // Get the number of clusters
  for( int d = 0; d < D; ++d ){
    if( cluster_kappa_temp[ d ] != 0 ){
      M_til += 1;
    }
  }
  
  // Update nuisance parameter eta_sA
  double eta_sA = R::rbeta( sA_temp + 1, n_lambda );
  
  // Update sA
  double b = b_sA - log( eta_sA ) ;
  double prob = ( a_sA + M_til - 1)/( n_lambda*b + a_sA + M_til - 1 );
  double pi_sA = rbinom( 1, 1, prob )[0];
  
  if( pi_sA == 1 ){
    sA_temp = rgamma(1, ( a_sA + M_til ) , 1/b )[0];
  }else{
    sA_temp = rgamma(1, ( a_sA + M_til - 1 ), 1/b )[0];
  }
  
  return sA_temp;
}



// Function :: Update Gamma
arma::mat update_Gamma( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat zeta_temp, arma::vec lambda_temp, arma::mat V_gamma, arma::vec gamma_0, arma::vec subject){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::mat G_inv( D, D );
  G_inv.zeros();
  arma::vec H( obs );
  H.zeros();
  
  // Get number of included random effects
  int D_lambda = sum( lambda_temp );
  
  // Set number of gamma parameters
  int num_gammas = D_lambda*(D_lambda-1)/2;
  
  // Set mean and variance for update based on included K
  arma::mat Vhat_gamma( num_gammas, num_gammas );
  arma::vec gamma_hat( num_gammas );
  arma::mat gamma_hat_inside( num_gammas, 1 );
  
  Vhat_gamma.zeros();
  gamma_hat.zeros();
  gamma_hat_inside.zeros();
  
  // Set up structure of Gamma
  arma::mat Gamma_update( D, D );
  Gamma_update.zeros();
  for( int d = 0; d < D; ++d ){
    Gamma_update( d, d ) = 1;
  }
  
  // Resize V_gamma and invert
  // Works under the assumption that the gammas are independent a priori (ie just takes diagonal terms)
  arma::mat V_gamma_red_inv( D_lambda*( D_lambda - 1 )/2, D_lambda*( D_lambda - 1 )/2 );
  V_gamma_red_inv.zeros();
  
  arma::vec gamma_0_red( D_lambda*( D_lambda - 1 )/2 );
  gamma_0_red.zeros();
  
  int count = 0;
  int count_red = 0; 
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        V_gamma_red_inv( count_red, count_red ) = 1/V_gamma( count, count );
        gamma_0_red[ count_red ] = gamma_0[ count ];
        count_red += 1;
      }
      count += 1; 
    }
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Make Zhat_lambda
  arma::mat Zhat_lambda( obs, D_lambda*(D_lambda-1)/2 );
  Zhat_lambda.zeros();
  Zhat_lambda =  help::make_Zhat_lambda( Z, K_temp, zeta_temp, lambda_temp, subject );
  
  // Update gammas currently included in the model ( based on lambda_temp )
  // Sum over all of the observations ij 
  for( int j = 0; j < obs; ++j ){
    Vhat_gamma += W[ j ]*(Zhat_lambda.row( j ).t()*Zhat_lambda.row( j )); 
    gamma_hat_inside +=  W[ j ]*Zhat_lambda.row( j ).t()*( H[ j ] -  sX.row( j )*beta_temp );
  }
  
  Vhat_gamma += V_gamma_red_inv;
  Vhat_gamma = inv( Vhat_gamma );
  
  gamma_hat_inside += V_gamma_red_inv*gamma_0_red;
  gamma_hat =  Vhat_gamma*gamma_hat_inside ;
  
  arma::mat gamma_elements( D_lambda*( D_lambda - 1 )/2, 1 );
  gamma_elements.zeros();
  gamma_elements = help::mvrnormArma( 1, gamma_elements, Vhat_gamma );
  
  // Append updated elements to the Gamma matrix appropriately 
  count_red = 0; 
  for( int m = 0; m < D - 1; ++m ){
    for( int l = m + 1; l < D; ++l ){
      if( lambda_temp[ l ] != 0 & lambda_temp[ m ] != 0 ){
        Gamma_update( l, m ) = gamma_elements[ count_red ];
        count_red += 1;
      }
    }
  }
  return Gamma_update;
}


// Function :: Update zeta
arma::mat update_zeta( arma::vec Y, arma::vec W, arma::vec beta_temp, arma::mat Ustar, arma::mat Xbar, arma::vec Ustar_dems, arma::vec xi_temp, arma::mat Z, arma::vec K_temp, arma::mat Gamma_temp, arma::vec subject_dems){
  int obs = Ustar.n_rows;
  int S = beta_temp.size();
  int N = subject_dems.size() - 1;
  arma::mat sX( obs, S );
  sX.zeros();
  int D = K_temp.size();
  arma::mat V_zeta( D, D );
  arma::vec mu_zeta( D );
  mu_zeta.zeros();
  arma::mat K_mat( D, D );
  K_mat.zeros();
  arma::mat I( D, D );
  I.zeros();
  arma::vec H( obs );
  H.zeros();
  arma::mat mu_zeta_inside( 1, D );
  
  arma::mat zeta_update( N, D );
  zeta_update.zeros();
  
  // Make K matrix
  for( int i = 0; i < D; ++i ){
    K_mat( i, i ) = K_temp[ i ];
  }
  
  // Make I
  for( int i = 0; i < D; ++i ){
    I( i, i ) = 1;
  }
  
  // Make h_ij = k_ij/w_ij, k_ij = y_ij - 1/2
  H = (Y - 0.5)/W;
  
  // Make sX
  sX =  help::make_sX( Ustar, xi_temp, Xbar, Ustar_dems );
  
  // Update for each individuals zeta_temp
  for( int n = 0; n < N; ++n ){
    
    // Pull the Z that are associated with subject i 
    arma::mat z_i = Z.rows( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Get the number of observations for subject i;
    int obs_i = z_i.n_rows;
    
    // Pull the w that are associated with subject i 
    arma::vec w_i = W.subvec( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    //Pull the h that are associated with subject i 
    arma::vec h_i = H.subvec( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Pull the sX that are associated with subject i
    arma::mat sX_i = sX.rows( subject_dems[ n ], subject_dems[ n + 1 ] - 1 );
    
    // Make V_zeta and mu_zeta 
    mu_zeta_inside.zeros();
    V_zeta.zeros();
    
    for( int j = 0; j < obs_i; ++j){
      V_zeta += w_i[ j ]*Gamma_temp.t()*K_mat*z_i.row( j ).t()*z_i.row( j )*K_mat*Gamma_temp;
      mu_zeta_inside += w_i[ j ]*( h_i[ j ] -  sX_i.row( j )*beta_temp )*z_i.row( j )*K_mat*Gamma_temp;
    }
    
    V_zeta += I; 
    
    V_zeta = inv( V_zeta );
    
    mu_zeta = (mu_zeta_inside*V_zeta).t();
    
    zeta_update.row( n ) = help::mvrnormArma( 1, mu_zeta, V_zeta ); 
    
  }
  
  return zeta_update;
}



}  // For namespace 'help'

// Function :: MCMC algorithm
// [[Rcpp::export]]
List bvsPGcpp(
    int iterations,             // Number of iterations
    int thin,                   // How often to thin to make the output less
    String prior,               // Doesn't have any use at this point but will
    bool DP_beta,               // Boolean to use DP prior for beta or not
    bool DP_kappa,              // Boolean to use DP prior for kappa or not 
    arma::vec Y,                // Y - Vector of outcomes. Indexed ij.
    arma::mat W,                // W - Matrix of MCMC samples for auxillary variables. Rows indexed ij. Columns indexed by MCMC sample
    arma::vec subject,          // subject - Vector that indicates which observations come from a subject. Elements are ij
    arma::vec subject_dems,     // subject_dems - Input vector of dimensions for each subject in the data. Element is equal to the starting indicies for corresponding n. Last term is the number of observations
    arma::mat Ustar,            // Ustar - Matrix of spline functions. Rows indexed by ij. Columns indexed by sum r_p over p. ( Should be a list if r_p != r_p' )
    arma::vec Ustar_dems,       // Ustar_dems - Input vector of dimensions for each spline function. Element is equal to starting indicies for corresponding p. Last term is number of columns. Ustar_dems[ 0 ] = 0 and length = P + 1
    arma::mat Xbar,             // Xbar - Matrix of barX. Rows indexed by ij. Columns indexed by 2P ( x1u, x1, x2u, x2,...xPu, xP )
    arma::mat Z,                // Z - Matrix of random covariates for each subject. Columns indexed by D. Rows indexed by ij
    IntegerVector random_avail, // random_avail - Vector including a list of random effects that are selectable ( ie not fixed in or out ). Indexed by D
    IntegerVector fixed_avail,  // fixed_avail - Vector including a list of fixed effects that are selectable ( ie not fixed in or out ). Indexed by S
    arma::mat beta,             // beta - Matrix of MCMC samples for beta. Rows indexed by beta_temp. Columns indexed by MCMC sample.
    arma::mat v,                // v - Matrix of MCMC samples for v. Rows indexed by v_temp. Columns indexed by MCMC sample.
    arma::mat xi,               // xi - Matrix of MCMC samples for parameter expansion for beta. Rows indexed by xi_temp. Columns indexed by MCMC sample.
    arma::mat mu,               // mu - Matrix of MCMC samples of means for each xi. Rows indexed by mu_tempp. Columns indexed by MCMC sample.
    arma::mat t2,               // t2 - Matrix of MCMC samples for t2. Rows indexed by t2_temp. Columns indexed by MCMC sample.
    arma::mat cluster,          // cluster - Matrix of MCMC samples for beta clusters. Rows indexed by cluster. Columns indexed by MCMC sample.
    arma::vec cluster_count,    // cluster_count - Vector of counts for each cluster 
    arma::mat cluster_beta,     // cluster_beta - Vector of beta values associated with each cluster 
    arma::vec vartheta,         // varthera - Vector of MCMC samplesfor concentration parameter H_0 ( for betas )
    arma::mat K,                // K - Matrix of MCMC samples for K. Rows indexed by K. Columns indexed by MCMC sample.
    arma::mat lambda,           // lambda -  Matrix of MCMC samples for lambda. Rows indexed by D. Columns indexed by MCMC sample
    arma::cube Gamma,           // Gamma - Array of MCMC samples for Gamma. x and y are indexed by Gamma_temp. z indexed by MCMC sample
    arma::cube zeta,            // zeta - Array of MCMC samples for zeta.  x and y indexed by zeta_temp. z indexed by MCMC sample
    arma::mat cluster_K,         // cluster_K - Matrix of MCMC samples for kappa clusters. Rows indexed by cluster. Columns indexed by MCMC sample.
    arma::vec cluster_count_K,   // cluster_count_K - Vector of counts for each kappa cluster 
    arma::mat cluster_kappa,     // cluster_kappa - Vector of kappa values associated with each cluster 
    arma::vec sA,                // sA - Vector of MCMC samplesfor concentration parameter W_0 ( for kappas )
    arma::mat V_gamma,           // V_gamma - Matrix of Gamma variance-covariance priors for MVT normal
    arma::vec gamma_0,           // gamma_0 - Vector of Gamma mean priors for MVT normal
    double m_star,               // m_star - double for folded normal proposal mean
    double v_star,               // v_star - double for folded normal proposal variance
    double m_0,                  // m_0 - double for folded normal prior mean
    double v_0,                  // v_0 - double for folded normal prior variance
    double a,                    // a - double for beta-binomial prior hyperparameter
    double b,                    // b - double for beta-binomial prior hyperparameter
    double a_0,                  // a_0 - double hyperparameter for t2
    double b_0,                  // b_0 - double hyperparameter for t2
    double a_g,                  // a_g - double hyperparameter for g
    double b_g,                  // b_g - double hyperparameter for g
    double a_vartheta,           // a_vartheta - double hyperparameter for vartheta
    double b_vartheta,           // b_vartheta - double hyperparameter for vartheta
    double a_sA,                 // a_sA - double hyperparameter for sA
    double b_sA                  // b_sA - double hyperparameter for sA
){
  
  // Initiate memory for List updates
  List return_rescaled( 2 );
  
  // Set temporary data to enable thinning
  arma::vec W_temp = W.col( 0 );            // W_temp - Vector of current auxillary parameters. Indexed by ij.
  arma::vec beta_temp = beta.col( 0 );      // beta_temp - Vector of coefficients for fixed effects (includes terms forced into model). Elements indexed by 3P. B*_p,B^0_p,B_p0,...
  arma::vec v_temp = v.col( 0 );            // v_temp - Vector of inclusion indicators for fixed effects (includes terms forced into model). Elements indexed by 3P.
  arma::vec xi_temp = xi.col( 0 );          // xi_temp - Vector of parameter expansion for beta. Elements indexed by sum r_p over p.
  arma::vec mu_temp = mu.col( 0 );          // mu_temp - Vector of means for each xi. Elements indexed by sum r_p over p.
  arma::vec t2_temp = t2.col( 0 );          // t2_temp - Vector of variances for fixed effects (includes terms forced into model). Elements indexed by 3P.
  arma::vec K_temp = K.col( 0 );            // K_temp - Vector of coefficients for random effects (includes terms forced into model). Elements indexed by D.
  arma::vec lambda_temp = lambda.col( 0 );  // lambda_temp - Indicator vector for random effect inclusion. Indexed by D.
  arma::mat Gamma_temp = Gamma.slice( 0 );  // Gamma_temp - Lower trianglular matrix for random effects. Columns and rows indexed by D.
  arma::mat zeta_temp = zeta.slice( 0 );    // zeta_temp - Matrix of random effects for each subject. Columns indexed by D. Rows indexed by i.
  arma::vec cluster_temp = cluster.col( 0 );// cluster_temp - Cluster assignement for betas. Indexed by S
  arma::vec cluster_count_temp = cluster_count; // cluster_count_temp - Vector of counts for beta clusters. Indexed by S. 
  arma::vec cluster_beta_temp = cluster_beta.col( 0 );    // cluster_beta_temp - Beta values for each cluster. Indexed by S.  
  arma::vec cluster_K_temp = cluster_K.col( 0 );         // cluster_K_temp - Cluster assignement for kappas. Indexed by D.
  double vartheta_temp = vartheta[ 0 ];                  // vartheta_temp - Concentration parameter for H_0
  arma::vec cluster_count_K_temp = cluster_count_K;       // cluster_count_K_temp - Vector of counts for kappa clusters. Indexed by D. 
  arma::vec cluster_kappa_temp = cluster_kappa.col( 0 );  // cluster_kappa_temp - Kappa values for each cluster. Indexed by D.  
  double sA_temp = sA[ 0 ];                               // sA_temp - Concentration parameter for W_0
  
  // Looping over the number of iterations specified by user
  for( int iter = 0; iter < iterations; ++iter ){
 
     // Update W
    W_temp = help::update_W( Ustar, xi_temp, Xbar, Ustar_dems, beta_temp, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject );
 
    // Update beta and v 
    if( DP_beta ){ // Update DP 
      
      // Update beta cluster assignments 

      List return_cluster_beta_DP( 4 );
      return_cluster_beta_DP = help::cluster_beta_cpp( cluster_temp, cluster_count_temp, cluster_beta_temp, vartheta_temp, beta_temp, v_temp, Y, W_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject, t2_temp, subject_dems, mu_temp  );

      cluster_temp = as<arma::vec>( return_cluster_beta_DP[ 0 ] );
      cluster_count_temp = as<arma::vec>( return_cluster_beta_DP[ 1 ] );
      cluster_beta_temp = as<arma::vec>( return_cluster_beta_DP[ 2 ] );
      beta_temp = as<arma::vec>( return_cluster_beta_DP[ 3 ] );
      
      // Between beta DP
      List return_between_beta_DP( 5 );
      return_between_beta_DP = help::between_step_beta_DP( cluster_temp, cluster_count_temp, cluster_beta_temp, vartheta_temp, Y, W_temp, beta_temp, v_temp, t2_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z,K_temp,fixed_avail,Gamma_temp,zeta_temp,subject,a,b );
      
      beta_temp = as<arma::vec>( return_between_beta_DP[ 0 ] );
      v_temp = as<arma::vec>( return_between_beta_DP[ 1 ] );
      cluster_temp = as<arma::vec>( return_between_beta_DP[ 2 ] );
      cluster_count_temp = as<arma::vec>( return_between_beta_DP[ 3 ] );
      cluster_beta_temp = as<arma::vec>( return_between_beta_DP[ 4 ] );
      
      // Within beta DP 
      List return_within_beta_DP( 2 );
      return_within_beta_DP = help::within_beta_DP( cluster_temp, cluster_count_temp, cluster_beta_temp, vartheta_temp, Y, W_temp, subject, beta_temp, t2_temp, v_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp);
      
      beta_temp = as<arma::vec>( return_within_beta_DP[ 0 ] );
      cluster_beta_temp = as<arma::vec>( return_within_beta_DP[ 1 ] );
      
      // Update vartheta 
      vartheta_temp = help::update_vartheta( vartheta_temp, v_temp, cluster_beta_temp, a_vartheta, b_vartheta );
      
      
    }else{ // Not DP but can take flexible inclusion priors in the future 
      
      // Between beta 
      if( prior == "BB" ){
        List return_between_beta = help::between_step_beta( Y, W_temp, beta_temp, v_temp, t2_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, fixed_avail, Gamma_temp, zeta_temp, subject, a, b );
        beta_temp = as<arma::vec>( return_between_beta[ 0 ] );
        v_temp = as<arma::vec>( return_between_beta[ 1 ] );
      }
      
      // Within beta 
      beta_temp = help::within_beta( Y, W_temp, subject, beta_temp, t2_temp, v_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp);
      
    }
    
    // Update xi
    xi_temp = help::update_xi( Y, W_temp, subject, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, zeta_temp, subject_dems, mu_temp);
    
    // Rescale
    return_rescaled = help::rescaled( beta_temp, xi_temp, Ustar_dems);
    beta_temp = as<arma::vec>( return_rescaled[ 0 ] );
    xi_temp = as<arma::vec>( return_rescaled[ 1 ] );
    
    // Update mu
    mu_temp = help::update_mu_rp( xi_temp );
    
    // Update t2 if not DP
    if( !DP_beta ){
      t2_temp = help::update_t2_s( beta_temp, a_0, b_0 );
    }
    
    // Update K and lambda 
    if( DP_kappa ){ // Update DP
      
      // Update kappa cluster assignments 
      List return_cluster_kappa_DP( 4 );
      return_cluster_kappa_DP = help::cluster_kappa_cpp( cluster_K_temp, cluster_count_K_temp, cluster_kappa_temp, sA_temp, beta_temp, v_temp, Y, W_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, lambda_temp, zeta_temp, subject, t2_temp, subject_dems, mu_temp, m_star, v_star, m_0, v_0  );
      
      cluster_K_temp = as<arma::vec>( return_cluster_kappa_DP[ 0 ] );
      cluster_count_K_temp = as<arma::vec>( return_cluster_kappa_DP[ 1 ] );
      cluster_kappa_temp = as<arma::vec>( return_cluster_kappa_DP[ 2 ] );
      K_temp = as<arma::vec>( return_cluster_kappa_DP[ 3 ] );
      
      // Between kappa DP 
      List return_between_kappa_DP( 5 );
      return_between_kappa_DP = help::between_step_K_DP( cluster_K_temp, cluster_count_K_temp, cluster_kappa_temp, sA_temp, Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, lambda_temp, random_avail, Gamma_temp, zeta_temp, subject, a, b, m_star, v_star, m_0, v_0 );
      
      K_temp = as<arma::vec>( return_between_kappa_DP[ 0 ] );
      lambda_temp = as<arma::vec>( return_between_kappa_DP[ 1 ] );
      cluster_K_temp = as<arma::vec>( return_between_kappa_DP[ 2 ] );
      cluster_count_K_temp = as<arma::vec>( return_between_kappa_DP[ 3 ] );
      cluster_kappa_temp = as<arma::vec>( return_between_kappa_DP[ 4 ] );
      
      // Within kappa DP 
      List return_within_kappa_DP( 2 );
      return_within_kappa_DP =  help::within_step_K_DP( cluster_K_temp, cluster_count_K_temp, cluster_kappa_temp, sA_temp, Y, subject, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, lambda_temp, Gamma_temp, zeta_temp, m_0, v_0 );
      
      K_temp = as<arma::vec>( return_within_kappa_DP[ 0 ] );
      cluster_kappa_temp = as<arma::vec>( return_within_kappa_DP[ 1 ] );
      
      // Update sA
      sA_temp = help::update_sA( sA_temp, lambda_temp, cluster_kappa_temp, a_sA, b_sA );
      
      
    }else{  // Not DP but can take flexible inclusion priors in the future 
      
      // Between K 
      List return_between_K( 2 );
      return_between_K = help::between_step_K( Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, lambda_temp, random_avail, Gamma_temp, zeta_temp, subject, a, b, m_star, v_star, m_0, v_0 );
      K_temp = as<arma::vec>( return_between_K[ 0 ] );
      lambda_temp = as<arma::vec>( return_between_K[ 1 ] );
      
      // Within K 
      K_temp = help::within_step_K( Y, subject, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, lambda_temp, Gamma_temp, zeta_temp, m_0, v_0 );
      
    }   
    //Update Gamma
    Gamma_temp = help::update_Gamma( Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, zeta_temp, lambda_temp, V_gamma, gamma_0, subject);
    
    // Update zeta
    zeta_temp = help::update_zeta( Y, W_temp, beta_temp, Ustar, Xbar, Ustar_dems, xi_temp, Z, K_temp, Gamma_temp, subject_dems );
    
    // Set the starting values for the next iteration
    if( ( iter + 1 ) % thin == 0 ){
      W.col( ( iter + 1 )/thin - 1 ) = W_temp;            // W_temp - Vector of current auxillary parameters. Indexed by ij.
      beta.col( ( iter + 1 )/thin - 1 ) = beta_temp;      // beta_temp - Vector of coefficients for fixed effects (includes terms forced into model). Elements indexed by 3P. B*_p,B^0_p,B_p0,...
      v.col( ( iter + 1 )/thin - 1 ) = v_temp;            // v_temp - Vector of inclusion indicators for fixed effects (includes terms forced into model). Elements indexed by 3P.
      xi.col( ( iter + 1 )/thin - 1 ) = xi_temp;          // xi_temp - Vector of parameter expansion for beta. Elements indexed by sum r_p over p.
      mu.col( ( iter + 1 )/thin - 1 ) = mu_temp;          // mu_temp - Vector of means for each xi. Elements indexed by sum r_p over p.
      t2.col( ( iter + 1 )/thin - 1 ) = t2_temp;          // t2_temp - Vector of variances for fixed effects (includes terms forced into model). Elements indexed by 3P.
      K.col( ( iter + 1 )/thin - 1 ) = K_temp;            // K_temp - Vector of coefficients for random effects (includes terms forced into model). Elements indexed by D.
      lambda.col( ( iter + 1 )/thin - 1 ) = lambda_temp;  // lambda_temp - Indicator vector for random effect inclusion. Indexed by D.
      Gamma.slice( ( iter + 1 )/thin - 1 ) = Gamma_temp;  // Gamma_temp - Lower trianglular matrix for random effects. Columns and rows indexed by D.
      zeta.slice( ( iter + 1 )/thin - 1 ) = zeta_temp;    // zeta_temp - Matrix of random effects for each subject. Columns indexed by D. Rows indexed by i.
      cluster.col( ( iter + 1 )/thin - 1 ) = cluster_temp;            // cluster_temp - Cluster assignement for betas. Indexed by S
      cluster_beta.col( ( iter + 1 )/thin - 1 ) = cluster_beta_temp;  // cluster_beta_temp - Beta values for each cluster. Indexed by S.  
      vartheta[ ( iter + 1 )/thin - 1 ] = vartheta_temp;                // vartheta - Concentration parameter for H
      cluster_K.col( ( iter + 1 )/thin - 1 ) = cluster_K_temp;            // cluster_K_temp - Cluster assignement for kappas. Indexed by D.
      cluster_kappa.col( ( iter + 1 )/thin - 1 ) = cluster_kappa_temp;  // cluster_kappa_temp - Kappa values for each cluster. Indexed by D.  
      sA[ ( iter + 1 )/thin - 1 ] = sA_temp;                              // sA - Concentration parameter for W
    }
    
    // Print out progress
    double printer = iter % 50;
    
    if( printer == 0 ){
      Rcpp::Rcout << "Iteration = " << iter << std::endl;
    }
  }
  
  // Return output
  List output( 16 );
  output[ 0 ] = W;
  output[ 1 ] = beta;
  output[ 2 ] = v;
  output[ 3 ] = xi;
  output[ 4 ] = mu;
  output[ 5 ] = t2;
  output[ 6 ] = K;
  output[ 7 ] = lambda;
  output[ 8 ] = Gamma;
  output[ 9 ] = zeta;
  output[ 10 ] = cluster;
  output[ 11 ] = cluster_beta; 
  output[ 12 ] = vartheta;
  output[ 13 ] = cluster_K;
  output[ 14 ] = cluster_kappa; 
  output[ 15 ] = sA; 
  
  return output ;
}

