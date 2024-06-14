# 
# Hyperparameters class
# 

class hyperparameters():

    def __init__(self, learning_rate, T_max, weight_decay, n_layers, hidden_channels, r_link, n_epochs, n_sims, pred_params, snap):

        # Learning rate
        self.learning_rate = learning_rate
        # Multiplicative factor of learning rate decay
        #self.gamma_lr = gamma_lr
        self.T_max = T_max
        # Weight decay
        self.weight_decay = weight_decay
        # Number of graph layers
        self.n_layers = n_layers
        # Hidden channels
        self.hidden_channels = hidden_channels
        # Linking radius
        self.r_link = r_link
        # Number of epochs
        self.n_epochs = n_epochs
        # Number of simulations considered
        self.n_sims = n_sims
        # Number of cosmo params to be predicted (Omega_m, sigma_8)
        self.pred_params = pred_params
        # Snapshot of the simulation, indicating redshift 4: z=3, 10: z=2, 14: z=1.5, 18: z=1, 24: z=0.5, 33: z=0
        self.snap = snap

# Istance
hparams = hyperparameters(
    learning_rate = 0.33e-04,
    T_max = 22,
    weight_decay = 1.e-03,
    n_layers = 2,
    r_link = 0.2,
    hidden_channels = 64,
    n_epochs = 300,
    n_sims = 2000,
    pred_params = 1,
    snap = 18
    )
