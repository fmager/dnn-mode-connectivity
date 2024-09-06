import jax.numpy as jnp
from jax import jit
from jax import random
from functools import partial
from typing import NamedTuple


@jit
def squared_exponential(tau, kappa, lengthscale):
    return kappa**2*jnp.exp(-0.5*tau**2/lengthscale**2)

@jit
def matern32(tau, kappa, lengthscale):
    return kappa**2*(1 + jnp.sqrt(3)*tau/lengthscale)*jnp.exp(-jnp.sqrt(3)*tau/lengthscale)

@jit
def compute_euclid_dist(X1, X2):
    return jnp.sqrt(jnp.sum((jnp.expand_dims(X1, 1) - jnp.expand_dims(X2, 0))**2, axis=-1))

@jit
def compute_mean_and_full_cov(K, k, Kstar, y, hyp):

  # Compute C matrix
    C = K + hyp.sigma**2*jnp.identity(len(y)) 

    # Compute cholesky decomp.
    L = jnp.linalg.cholesky(C)
    B = jnp.linalg.solve(L, k.T)

    # computer mean and Sigma
    mu = B.T@jnp.linalg.solve(L, y)
    Sigma = Kstar - B.T@B

    return mu, Sigma

@jit
def compute_mean_and_var(K, k, Kstar_diag, y, hyp):

  # Compute C matrix
    C = K + hyp.sigma**2*jnp.identity(len(y)) 

    # Compute cholesky decomp.
    L = jnp.linalg.cholesky(C)
    B = jnp.linalg.solve(L, k.T)

    # computer mean and diagonal of Sigma
    mu = B.T@jnp.linalg.solve(L, y)
    Sigma = Kstar_diag - jnp.sum(B**2, axis=0)

    return mu, Sigma


class Hyperparameters(NamedTuple):
    kappa:       float
    lengthscale: float
    sigma:      float

    def to_flat(self):
        return jnp.log(jnp.array([self.kappa, self.lengthscale, self.sigma]))
    
    def from_flat(self, theta):
        self.kappa, self.lengthscale, self.sigma = jnp.exp(theta)
        return self
    


class StationaryIsotropicKernel(object):

    def __init__(self, kernel_fun):
        """
            the argument kernel_fun must be a function of three arguments kernel_fun(||tau||, kappa, lengthscale), e.g. 
            squared_exponential = lambda tau, kappa, lengthscale: kappa**2*np.exp(-0.5*tau**2/lengthscale**2)
        """
        self.kernel_fun = kernel_fun

    def contruct_kernel(self, X1, X2, hyperparams, jitter=1e-8):
        """ compute and returns the NxM kernel matrix between the two sets of input X1 (shape NxD) and X2 (MxD) using the stationary and isotropic covariance function specified by self.kernel_fun
    
        arguments:
            X1              -- NxD matrix
            X2              -- MxD matrix
            kappa           -- magnitude (positive scalar)
            lengthscale     -- characteristic lengthscale (positive scalar)
            jitter          -- non-negative scalar
        
        returns
            K               -- NxM matrix    
        """

        # extract dimensions 
        N, M = X1.shape[0], X2.shape[0]

        # compute all the pairwise distances efficiently
        dists = compute_euclid_dist(X1, X2)

        # squared exponential covariance function
        K = self.kernel_fun(dists, hyperparams.kappa, hyperparams.lengthscale)
        
        # add jitter to diagonal for numerical stability
        if len(X1) == len(X2) and jnp.allclose(X1, X2):
            K = K + jitter*jnp.identity(len(X1))
        
        assert K.shape == (N, M), f"The shape of K appears wrong. Expected shape ({N}, {M}), but the actual shape was {K.shape}. Please check your code. "
        return K
    
    def marginal_variance(self, X1, hyperparams):
        return self.kernel_fun(jnp.zeros(len(X1)), hyperparams.kappa, hyperparams.lengthscale)

class SquaredExponentialKernel(StationaryIsotropicKernel):
    def __init__(self):
        super().__init__(squared_exponential)

class Matern32Kernel(StationaryIsotropicKernel):
    def __init__(self):
        super().__init__(matern32)



def generate_samples(key, m, K, num_samples, jitter=1e-6):
    """ returns M samples from an Gaussian process with mean m and kernel matrix K. The function generates num_samples of z ~ N(0, I) and transforms them into f  ~ N(m, K) via the Cholesky factorization.

    
    arguments:
        m                -- mean vector (shape (N,))
        K                -- kernel matrix (shape NxN)
        num_samples      -- number of samples to generate (positive integer)
        jitter           -- amount of jitter (non-negative scalar)
    
    returns 
        f_samples        -- a numpy matrix containing the samples of f (shape N x num_samples)
    """
    
    N = len(K)
    L = jnp.linalg.cholesky(K + jitter*jnp.identity(N))
    zs = random.normal(key, shape=(len(K), num_samples))
    f_samples = m[:, None] + jnp.dot(L, zs)
    
    # sanity check of dimensions
    assert f_samples.shape == (len(K), num_samples), f"The shape of f_samples appears wrong. Expected shape ({len(K)}, {num_samples}), but the actual shape was {f_samples.shape}. Please check your code. "
    return f_samples



class GaussianProcessRegression(object):

    def __init__(self, X, y, kernel, hyperparameters, jitter=1e-8):
        """  
        Arguments:
            X                -- NxD input points
            y                -- Nx1 observed values 
            kernel           -- must be instance of the StationaryIsotropicKernel class
            jitter           -- non-negative scaler
        """
        self.X = jnp.array(X, dtype=jnp.float64)
        self.y = jnp.array(y, dtype=jnp.float64)
        self.N = len(X)
        self.kernel = kernel
        self.jitter = jitter
        self.hyperparameters = hyperparameters

    def plot_prior1d(self, ax, Xstar, color='b', color_samples='b', title="", num_samples=0, key=None, label=None, plot_distribution=True):
        return self.plot1d(ax, Xstar, color=color, color_samples=color_samples, title=title, num_samples=num_samples, key=key, prior=True, label=label, plot_distribution=plot_distribution)

    def plot_posterior1d(self, ax, Xstar, color='b', color_samples='b', title="", num_samples=0, key=None, label=None, plot_distribution=True):
        return self.plot1d(ax, Xstar, color=color, color_samples=color_samples, title=title, num_samples=num_samples, key=key, prior=False, label=label, plot_distribution=plot_distribution)


    def plot1d(self, ax, Xstar, color='r', color_samples='b', title="", num_samples=0, key=None, prior=False, label=None, plot_distribution=True):
    
        if plot_distribution:
            # make predictions
            if prior:
                mu, Sigma_diag = self.prior_y(Xstar, full_cov=False)
            else:
                mu, Sigma_diag = self.predict_y(Xstar, full_cov=False)
            mean = mu.ravel()
            std = jnp.sqrt(Sigma_diag)

        # plot distribution
            ax.plot(Xstar, mean, color=color, alpha=0.75)
            ax.plot(Xstar, mean + 1.96*std, color=color, linestyle='--', alpha=0.2)
            ax.plot(Xstar, mean - 1.96*std, color=color, linestyle='--', alpha=0.2)

            if label is None:
                label = 'Mean + 95% interval'
            ax.fill_between(Xstar.ravel(), mean - 2*std, mean + 2*std, color=color, alpha=0.2, label=label)
    
        # generate samples
        if num_samples > 0:
            if prior:
                fs = self.prior_samples(key, Xstar, num_samples)
            else:
                fs = self.posterior_samples(key, Xstar, num_samples)
            ax.plot(Xstar, fs[:,0], color=color_samples, alpha=.2, label="$f(x)$ samples", linewidth=1.3)
            ax.plot(Xstar, fs[:, 1:], color=color_samples, alpha=.2, linewidth=1.3)

        
        ax.set_title(title, fontweight='bold')


    def posterior_samples(self, key, Xstar, num_samples):
        """
            generate samples from the posterior p(f^*|y, x^*) for each of the inputs in Xstar

            Arguments:
                Xstar            -- PxD prediction points
        
            returns:
                f_samples        -- numpy array of (P, num_samples) containing num_samples for each of the P inputs in Xstar
        """
        mu, Sigma = self.predict_f(Xstar, full_cov=True)
        f_samples = generate_samples(key, mu.ravel(), Sigma, num_samples)

        return f_samples
    
    def prior_samples(self, key, Xstar, num_samples):
        mu, Sigma = self.prior_f(Xstar, full_cov=True)
        f_samples = generate_samples(key, mu.ravel(), Sigma, num_samples)

        return f_samples
    

    def prior_f(self, Xstar, full_cov=False):

        mu = jnp.zeros((len(Xstar), 1))
        
        if full_cov:
            Kstar = self.kernel.contruct_kernel(Xstar, Xstar, self.hyperparameters, jitter=self.jitter)
        else:
            raise NotImplementedError
        
        return mu, Kstar
    
    def prior_y(self, Xstar, full_cov=False):

        mu = jnp.zeros((len(Xstar), 1))
        
        if full_cov:
            Kstar = self.kernel.contruct_kernel(Xstar, Xstar, self.hyperparameters, jitter=self.jitter)
        else:
            Kstar = self.kernel.marginal_variance(Xstar, self.hyperparameters)
        
        return mu, Kstar

        
    def predict_y(self, Xstar, full_cov=False):
        """ returns the posterior distribution of y^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        # prepare relevant matrices
        mu, Sigma = self.predict_f(Xstar, full_cov)
        if full_cov:
            Sigma = Sigma + self.hyperparameters.sigma**2 * jnp.identity(len(mu))
        else:
            Sigma = Sigma + self.hyperparameters.sigma**2

        return mu, Sigma

    def predict_f(self, Xstar, full_cov=False):
        """ returns the posterior distribution of f^* evaluated at each of the points in x^* conditioned on (X, y)
        
        Arguments:
        Xstar            -- PxD prediction points
        
        returns:
        mu               -- Px1 mean vector
        Sigma            -- PxP covariance matrix
        """

        # prepare relevant matrices
        k = self.kernel.contruct_kernel(Xstar, self.X, self.hyperparameters, jitter=self.jitter)
        K = self.kernel.contruct_kernel(self.X, self.X, self.hyperparameters, jitter=self.jitter)
        Kstar = self.kernel.contruct_kernel(Xstar, Xstar, self.hyperparameters, jitter=self.jitter)
        
        # Compute C matrix
        C = K + self.hyperparameters.sigma**2*jnp.identity(len(self.X)) 

        # Compute cholesky decomp.
        L = jnp.linalg.cholesky(C)
        B = jnp.linalg.solve(L, k.T)

        # computer mean and Sigma
        mu = B.T@jnp.linalg.solve(L, self.y)

        if full_cov:
            mu, Sigma = compute_mean_and_full_cov(K, k, Kstar, self.y, self.hyperparameters)
        else:
            Kstar_diag = self.kernel.marginal_variance(Xstar, self.hyperparameters)
            mu, Sigma = compute_mean_and_var(K, k, Kstar_diag, self.y, self.hyperparameters)

        return mu, Sigma
    
    def log_marginal_likelihood(self, theta):
        """ 
            evaluate the log marginal likelihood p(y) given the hyperparaemters 

            Arguments:
            kappa       -- positive scalar 
            lengthscale -- positive scalar
            sigma       -- positive scalar
            """

        h = Hyperparameters(*jnp.exp(theta))#(kappa=theta[0], lengthscale=theta[1], sigma=theta[2])
        K = self.kernel.contruct_kernel(self.X, self.X, h)
        C = K + h.sigma**2*jnp.identity(self.N)

        # compute Cholesky decomposition
        L = jnp.linalg.cholesky(C)
        v = jnp.linalg.solve(L, self.y)

        # compute log marginal likelihood
        logdet_term = jnp.sum(jnp.log(jnp.diag(L)))
        quad_term =  0.5*jnp.sum(v**2)
        const_term = -0.5*self.N*jnp.log(2*jnp.pi)

        return const_term - logdet_term - quad_term

