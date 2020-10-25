import numpy as np
from scipy.linalg import svd

def stacked_lstsq(L, b, rcond=1e-10):
    """
    Solve L x = b, via SVD least squares cutting of small singular values
    L is an array of shape (M, N) and b of shape (..., M).
    Returns x of shape (..., N)
    """
    u, s, v = np.linalg.svd(L, full_matrices=False)
    s_max = s.max(axis=-1, keepdims=True)
    s_min = rcond*s_max
    inv_s = np.zeros_like(s)
    inv_s[s >= s_min] = 1/s[s>=s_min]
    x = np.einsum('ji,...j-> ...i', v, inv_s * np.einsum('ji,...j->...i', u, b))
    return x



def probabilistic_pca(data, q, method="mle", epsilon=1e-6):
    
    # initialization
    m = data.shape[1]
    
    if method == "mle":

        # empirical mean
        mu_hat = np.mean(data, axis=0)

        # empirical covariance
        xi = data - mu_hat[np.newaxis, :]
        sigma_hat = np.mean(np.einsum("...i, ...j -> ...ij", xi, xi), axis=0)

        u, lam, _ = svd(sigma_hat)

        # sufficient statistics
        if q < m:
            variance_hat = np.mean(lam[q:])  # variance lost in the projection
            lam[q:] = variance_hat           # principal eigenvalues unchanged
            u_q = u[:, :q]                   # projector to principal space
        elif q == m:
            variance_hat = 0
            u_q = u

        # compute square root via svd
        A = (np.eye(m) * lam - np.eye(m) * variance_hat)
        v, d, v_inv = svd(A)
        A_sq = v_inv @ (np.eye(m) * d**(1/2)) @ v
        
        # ML estimator of weight matrix
        W = u_q.T @ A_sq
    
    # optimize for the variance sigma and weight matrix
    if method == "em":

        def expected_log_likelihood(exp_eta, exp_eta_cov, var, W):
#             xi = data - mu[np.newaxis, :] # gotten from outside namescope
            
            out = N * m /2 * np.log(var)
            out += np.sum(np.einsum("...ii -> ...", exp_eta_cov))/2
            out += np.sum(np.einsum("...i, ...i -> ...", xi, xi))/2/var
            out -= np.sum(np.einsum("...i, ...i -> ...", exp_eta, \
                    np.einsum("ij, ...i -> ...j", W, xi)))/var
            w_cov =  np.einsum("ij, ...jk -> ...ik", W, exp_eta_cov)
            ww_cov = np.einsum("ji, ...jk -> ...ik", W, w_cov)
            trace = np.einsum("...ii -> ...", ww_cov)
            out += np.sum(trace)/2/var
            return -out
        
        # estimate mu from data
        mu_hat = np.mean(data, axis=0)
        xi = data - mu_hat[np.newaxis, :]
        
        expectation_last = -np.inf    # start expectation at the lowest possible value
        best_expectation = -np.inf    # set a record for the space search
    
        max_iter = 1000               # set a maximum iteration limit
        n_try = 1                     # explore the landscape to search for global maxima
        # as it turns out, one iteration is fine.
        
        for epoch in range(n_try):
            # initialize latent variable at a random point in the parameter space
            W_tilde = np.random.normal(0, 1, size=(m, q))
            var_tilde = np.random.normal(0, 1)**2
            for i in range(1000):
                # Expectation step
                M = W_tilde.T @ W_tilde + np.eye(q) * var_tilde   # (q, q) matrix
                M_inv = np.linalg.inv(M)
                exp_eta = np.einsum("ij, ...j -> ...i", M_inv, \
                        np.einsum("ji, ...j -> ...i",  W_tilde, xi))
                exp_eta_cov = (M_inv * var_tilde)[np.newaxis, :, :] + \
                        np.einsum("...i, ...j -> ...ij", exp_eta, exp_eta)

                # Maximization step
                cov_norm = np.linalg.inv(np.sum(exp_eta_cov, axis=0))
                W_tilde = np.sum(np.einsum("...i, ...j -> ...ij", \
                        xi, exp_eta), axis=0) @ cov_norm
                var_tilde = np.sum(xi**2)
                var_tilde -= 2 * np.sum(np.einsum("...i, ...i -> ...",\
                        exp_eta, np.einsum("ij, ...i -> ...j", W_tilde, xi)))
                var_tilde += np.sum(np.einsum("...ii", \
                        np.einsum("...ij, jk -> ...ik", exp_eta_cov , W_tilde.T @ W_tilde)))
                var_tilde /= m * N


                expectation = expected_log_likelihood(exp_eta, exp_eta_cov, var_tilde, W_tilde)
                if expectation <= expectation_last + epsilon:
                    break
                expectation_last = expectation
            
            print(expectation)
            if expectation > best_expectation:
                # update our best parameters
                W = W_tilde
                var = var_tilde
                best_expectation = expectation

        
        # W span the principal subspace: it's ok that the principal component are not orthogonal
        u = W

    def sample_data(N):
        eta = np.random.multivariate_normal(mean=[0]*q, cov=np.eye(q), size=N)
        return np.einsum("ij, ...j -> ...i", W, eta) + mu_hat[np.newaxis, :]
        
    
    def principal_component():
        return np.einsum("ji, ...j -> ...i", W, (data - mu_hat[np.newaxis, :]))
        
    return sample_data, principal_component, u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Dataset
    N = 5000
    m = 2    # data dimension
    q = 1    # latent dimension for 
    variance = 0.5

    plt.style.use("dark_background")
    plt.figure()
    true_mean = [0, 0]
    true_cov = [[8, 1], [1, 1]]  # positive semi-definite and symettric matrix
    data = np.random.multivariate_normal(true_mean, true_cov, size=N)
    plt.hist2d(data[:, 0], data[:, 1], cmap="hot", bins=50)
    plt.axis("equal")
    plt.xlabel(r"x_0")
    plt.ylabel(r"x_1")
    plt.title("Dataset drawn from 2D Multivariate Gaussian")

    # Expectatation maximization output
    ppca = probabilistic_pca(data, 2, "em")

    new_sample = ppca[0](N)
    plt.figure()
    plt.hist2d(new_sample[:, 0], new_sample[:, 1], cmap="hot", bins=20)
    plt.axis("equal")
    plt.xlabel(r"x_0")
    plt.ylabel(r"x_1")
    plt.title("Reconstructed samples with EM")

    u1 = ppca[2][:, 0]
    u2 = ppca[2][:, 1]
    plt.figure()
    plt.hist2d(data[:, 0], data[:, 1], cmap="hot", bins=50)
    plt.axis("equal")
    plt.xlabel(r"x_0")
    plt.ylabel(r"x_1")
    plt.title("Principal axis with EM")
    plt.arrow(0, 0, 2*u1[0], 2*u1[1], color="w", width=0.5)
    plt.arrow(0, 0, 2*u2[0], 2*u2[1], color="w", width=0.2)

    # Maximum likelihood output
    ppca = probabilistic_pca(data, 2, "mle")

    new_sample = ppca[0](N)
    plt.figure()
    plt.hist2d(new_sample[:, 0], new_sample[:, 1], cmap="hot", bins=20)
    plt.axis("equal")
    plt.xlabel(r"x_0")
    plt.ylabel(r"x_1")
    plt.title("Reconstructed samples with MLE")

    u1 = ppca[2][:, 0]
    u2 = ppca[2][:, 1]
    plt.figure()
    plt.hist2d(data[:, 0], data[:, 1], cmap="hot", bins=50)
    plt.axis("equal")
    plt.xlabel(r"x_0")
    plt.ylabel(r"x_1")
    plt.title("Principal axis with MLE")
    plt.arrow(0, 0, 2*u1[0], 2*u1[1], color="w", width=0.5)
    plt.arrow(0, 0, 2*u2[0], 2*u2[1], color="w", width=0.2)

    plt.show()

    

