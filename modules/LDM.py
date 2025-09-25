import torch 
import torch.nn as nn

class LDM(nn.Module):
    def __init__(self, **kwargs):
        '''
        Initializes the linear dynamical model (LDM) object
        '''
        
        super(LDM, self).__init__()

        self.n_x = kwargs.pop('n_x', None)
        self.n_a = kwargs.pop('n_a', None)

        # Initializer for identity matrix and zeros matrix
        self.eye_init = lambda shape, dtype=torch.float32: torch.eye(*shape, dtype=dtype)
        self.zeros_init = lambda shape, dtype=torch.float32: torch.zeros(*shape, dtype=dtype)
        self.ones_init = lambda shape, dtype=torch.float32: torch.ones(*shape, dtype=dtype)

        # Get initial values for SS parameters
        self.A = kwargs.pop('A', self.eye_init((self.n_x, self.n_x), dtype=torch.float32).unsqueeze(dim=0)).type(torch.FloatTensor)
        self.C = kwargs.pop('C', self.eye_init((self.n_a, self.n_x), dtype=torch.float32).unsqueeze(dim=0)).type(torch.FloatTensor)
        
        # Get KF initial conditions
        self.Sigma_0 = kwargs.pop('Sigma_0', self.eye_init((self.n_x, self.n_x), dtype=torch.float32)).type(torch.FloatTensor)
        self.x_0 = kwargs.pop('x_0', self.zeros_init((self.n_x, ), dtype=torch.float32)).type(torch.FloatTensor)

        # Learnable noise covariance matrices in diagonal form
        self.Q_log_diag = kwargs.pop('Q_log_diag', self.eye_init((self.n_x, self.n_x), dtype=torch.float32)).type(torch.FloatTensor)
        self.R_log_diag = kwargs.pop('R_log_diag', self.eye_init((self.n_a, self.n_a), dtype=torch.float32)).type(torch.FloatTensor)
        
        self.converge_t = torch.inf
        self.apply_normalize_A = kwargs.pop('apply_normalize_A', False)

        # Register trainable parameters to module
        self._register_params()


    def _register_params(self):
        '''
        Registers the learnable LDM parameters
        '''

        # Register SS parameters
        self.A = torch.nn.Parameter(self.A, requires_grad=True)
        self.C = torch.nn.Parameter(self.C, requires_grad=True)

        self.Q_log_diag = torch.nn.Parameter(self.Q_log_diag, requires_grad=True)
        self.R_log_diag = torch.nn.Parameter(self.R_log_diag, requires_grad=True)

        # These parameters are not learnable 
        self.x_0 = torch.nn.Parameter(self.x_0, requires_grad=False)
        self.Sigma_0 = torch.nn.Parameter(self.Sigma_0, requires_grad=False)


    def _get_covariance_matrices(self):
        '''
        Computes the diagonal noise covariance matrices from their logarithms
        '''

        Q = torch.diag(torch.exp(self.Q_log_diag))
        R = torch.diag(torch.exp(self.R_log_diag))
        return Q, R
    
    @staticmethod
    def has_converged(prev_matrix, current_matrix, percentage_tol=1e-6):
        return (current_matrix - prev_matrix).abs().mean() / current_matrix.abs().mean() < percentage_tol

    def compute_forwards(self, a, mask=None):
        '''
        Performs filtering of multiscale or single scale or modality-specific latent factors and their estimation error covariances

        a: torch.Tensor, observations to the LDM module, (num_seq, num_steps, n_a)
        mask: torch.Tensor, mask tensor denoting whether  observations 'a' are available or not, (num_seq, num_steps)
        '''    

        if mask is None:
            mask = torch.ones(a.shape[:-1], dtype=torch.float32, device=a.device)
        
        num_seq, num_steps, _ = a.shape

        # We can mask either all dimensions or if provided, selected dimensions
        if len(mask.shape) != len(a.shape):
            mask = mask[:, :, None] # (num_seq, num_steps, 1)

        # ----------------------------------------------------------------------------
        # To make sure we are not accidentally using the real outputs in the steps with missing values, set them to 0.
        a_masked = torch.mul(a, mask) # (num_seq, num_steps, n_a) x (num_seq, num_steps, 1)
        # ----------------------------------------------------------------------------
        
        # ----------------------------------------------------------------------------
        # Initialize x_0 and Sigma_0 
        x_0 = self.x_0[None, :].repeat(num_seq, 1) # (num_seq, n_x)
        Sigma_0 = self.Sigma_0[None, :, :].repeat(num_seq, 1, 1) # (num_seq, n_x, n_x)

        x_pred = x_0 # (num_seq, n_x)
        Sigma_pred = Sigma_0 # (num_seq, n_x, n_x)
        # ----------------------------------------------------------------------------
        
        # ----------------------------------------------------------------------------
        # Create empty arrays for filtered and predicted state estimates
        x_pred_all = [] 
        x_filter_all = [] 

        # Create empty arrays for filtered and predicted state error covariance
        Sigma_pred_all = [] 
        Sigma_filter_all = [] 
        
        # Get covariance matrices
        Q, R = self._get_covariance_matrices()
        
        converged = False
        self.converge_t = torch.inf
        # ----------------------------------------------------------------------------
        for t in range(num_steps):
            C = self.C[None, :, :]
            A = self.A[None, :, :]

            # Obtain residual
            a_pred = (C @ x_pred[:, :, None]).squeeze(dim=-1) # (num_seq, n_a)
            r = a_masked[:, t, :] - a_pred # (num_seq, n_a)

            # project system uncertainty into measurement space, get Kalman gain
            R_rep = torch.tile(R, (num_seq, 1, 1))

            if not converged or self.training:
                if self.training or len(Sigma_pred_all) < 2 or not self.has_converged(Sigma_pred_all[-2], Sigma_pred_all[-1], percentage_tol=1e-3):
                    S = C @ Sigma_pred @ torch.permute(C, (0, 2, 1)) + R_rep # (num_seq, n_a, n_a)
                    L = torch.linalg.cholesky(S)
                    S_inv = torch.cholesky_inverse(L) # (num_seq, n_a, n_a)
                else:
                    converged = True
                    self.converge_t = t

            K = Sigma_pred @ torch.permute(C, (0, 2, 1)) @ S_inv # (num_seq, n_x, n_a)

            # Mask the Kalman gain for missing time-steps
            K = torch.mul(K, mask[:, t:t+1, :])  # (num_seq , n_x, n_a) x (num_seq, 1, n_a or 1)

            # Get filtered x and Sigma
            x_t = x_pred + (K @ r[:, :, None]).squeeze(dim=-1) # (num_seq, n_x)
            I_KC = torch.eye(self.n_x, dtype=torch.float32, device=x_0.device) - K @ C # (num_seq, n_x, n_x)
            Sigma_t = I_KC @ Sigma_pred @ torch.permute(I_KC, (0, 2, 1)) + K @ R_rep @ torch.permute(K, (0, 2, 1)) # (num_seq, n_x, n_x)

            # Prediction 
            x_pred = (A @ x_t[:, :, None]).squeeze(dim=-1) # (num_seq, n_x, n_x) x (num_seq, n_x, 1) --> (num_seq, n_x, 1) --> (num_seq, n_x)
            Sigma_pred = A @ Sigma_t @ torch.permute(A, (0, 2, 1)) + Q # (num_seq, n_x, n_x) x (num_seq, n_x, n_x) x (num_seq, n_x, n_x) --> (num_seq, n_x, n_x)

            x_pred_all.append(x_pred.unsqueeze(dim=0))
            x_filter_all.append(x_t.unsqueeze(dim=0))

            Sigma_pred_all.append(Sigma_pred.unsqueeze(dim=0))
            Sigma_filter_all.append(Sigma_t.unsqueeze(dim=0))

        # Convert list of tensors to matrices
        x_pred_all = torch.cat(x_pred_all, dim=0)
        x_filter_all = torch.cat(x_filter_all, dim=0)
        Sigma_pred_all = torch.cat(Sigma_pred_all, dim=0)
        Sigma_filter_all = torch.cat(Sigma_filter_all, dim=0)

        return x_pred_all, x_filter_all, Sigma_pred_all, Sigma_filter_all, Q, R

        
    def filter(self, a, mask=None):
        '''
        Returns filtered multiscale or single scale or modality-specific latent factors

        a: torch.Tensor, observations to the LDM module, (num_seq, num_steps, n_a)
        mask: torch.Tensor, mask tensor denoting whether  observations 'a' are available or not, (num_seq, num_steps)
        '''
        
        x_pred_all, x_filter_all, Sigma_pred_all, Sigma_filter_all, Q, R = self.compute_forwards(a=a, mask=mask)

        # Swab num_seq and num_steps dimensions
        x_pred_all = torch.permute(x_pred_all, (1, 0, 2))
        x_filter_all = torch.permute(x_filter_all, (1, 0, 2))

        Sigma_pred_all = torch.permute(Sigma_pred_all, (1, 0, 2, 3))
        Sigma_filter_all = torch.permute(Sigma_filter_all, (1, 0, 2, 3))

        return x_pred_all, x_filter_all, None, Sigma_pred_all, Sigma_filter_all, None


    def compute_backwards(self, forward_states):
        '''
        Performs backward pass on the filtered and predicted latent factors to obtain smoothed latent factors

        forward_states: tuple, contains predicted and filtered latent factors and their estimation error covariances, as well as noise covariance matrices
        '''

        # Get filtered estimates 
        x_pred_all, x_filter_all, Sigma_pred_all, Sigma_filter_all, Q, R = forward_states
        num_steps = x_pred_all.shape[0] 

        # Create empty arrays for smoothed states and error covariances
        x_smooth_all = []  
        Sigma_smooth_all = [] 

        x_smooth_all.append(x_filter_all[-1, :, :].unsqueeze(dim=0))
        Sigma_smooth_all.append(Sigma_filter_all[-1, :, :].unsqueeze(dim=0))

        # Initialize iterable parameter
        x_smooth = x_filter_all[-1, :, :]
        Sigma_smooth = Sigma_filter_all[-1, :, :, :]

        if num_steps >= 2:
            L = torch.linalg.cholesky(Sigma_pred_all[-2, :, :, :])
            Sigma_pred_inv = torch.cholesky_inverse(L)

        for t in range(num_steps-2, -1, -1): # iterate loop over reverse time: T-2, T-3, ..., 0, where last time-step is T-1
            if t <= self.converge_t:
                L = torch.linalg.cholesky(Sigma_pred_all[t, :, :, :])
                Sigma_pred_inv = torch.cholesky_inverse(L)
            J_t = Sigma_filter_all[t, :, :, :] @ torch.permute(self.A[None, :, :], (0, 2, 1)) @ Sigma_pred_inv # (num_seq, n_x, n_x) x (num_seq, n_x, n_x) x (num_seq, n_x, n_x)

            x_smooth = x_filter_all[t, :, :] + (J_t @ (x_smooth - x_pred_all[t, :, :]).unsqueeze(dim=-1)).squeeze(dim=-1) # (num_seq, n_x) + (num_seq, n_x, n_x) x (num_seq, n_x)

            Sigma_smooth = Sigma_filter_all[t, :, :, :] + J_t @ (Sigma_smooth - Sigma_pred_all[t, :, :, :]) @ torch.permute(J_t, (0, 2, 1)) # (num_seq, n_x, n_x)

            x_smooth_all.append(x_smooth.unsqueeze(dim=0))
            Sigma_smooth_all.append(Sigma_smooth.unsqueeze(dim=0))
        
        # Convert list of tensors to matrices
        x_smooth_all = torch.cat(x_smooth_all, dim=0)
        Sigma_smooth_all = torch.cat(Sigma_smooth_all, dim=0)

        x_smooth_all = torch.flip(x_smooth_all, [0]) # it was reversed in time
        Sigma_smooth_all = torch.flip(Sigma_smooth_all, [0]) # it was reversed in time

        return x_pred_all, x_filter_all, x_smooth_all, Sigma_pred_all, Sigma_filter_all, Sigma_smooth_all, 


    def smooth(self, a, mask=None):
        '''
        Returns filtered and smoothed multiscale or single scale or modality-specific latent factors

        a: torch.Tensor, observations to the LDM module, (num_seq, num_steps, n_a)
        mask: torch.Tensor, mask tensor denoting whether  observations 'a' are available or not, (num_seq, num_steps)
        '''

        x_pred_all, x_filt_all, x_smooth_all, Sigma_pred_all, Sigma_filt_all, Sigma_smooth_all = self.compute_backwards(self.compute_forwards(a=a, mask=mask))

        # Swab num_seq and num_steps dimensions
        x_pred_all = torch.permute(x_pred_all, (1, 0, 2))
        x_filt_all = torch.permute(x_filt_all, (1, 0, 2))
        x_smooth_all = torch.permute(x_smooth_all, (1, 0, 2))

        Sigma_pred_all = torch.permute(Sigma_pred_all, (1, 0, 2, 3))
        Sigma_filt_all = torch.permute(Sigma_filt_all, (1, 0, 2, 3))
        Sigma_smooth_all = torch.permute(Sigma_smooth_all, (1, 0, 2, 3))

        return x_pred_all, x_filt_all, x_smooth_all, Sigma_pred_all, Sigma_filt_all, Sigma_smooth_all

    def normalize_A(self, sigma_max=1.5):
        '''
        Normalizes the A matrix's singular values to prevent overshoots and bad trainings

        sigma_max: Maximum allowed singular value, defaults to 1.5
        '''
        with torch.no_grad():
            for i in range(self.A.shape[0]):
                u, sigma, v = torch.svd(self.A[i, ...])
                sigma_max_this = sigma.max()
                if sigma_max_this > sigma_max:
                    self.A[i, ...] *= (sigma_max / sigma_max_this)

    def forward(self, a, mask=None, do_smoothing=False): 
        '''
        Default forward function of LDM

        a: torch.Tensor, observations to the LDM module, (num_seq, num_steps, n_a)
        mask: torch.Tensor, mask tensor denoting whether  observations 'a' are available or not, (num_seq, num_steps)
        do_smoothing: bool,  whether to perform smoothing or not. if False, x_smooth and Sigma_smooth are returned None.
        '''
        
        # to prevent overshoots
        if self.apply_normalize_A:
            self.normalize_A()
        
        if do_smoothing:
            return self.smooth(a=a, mask=mask)
        else:
            return self.filter(a=a, mask=mask)




