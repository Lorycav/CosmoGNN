import time, datetime, psutil

from Source.metalayer import *
from Source.plotting import *
from Source.training import *
from Source.load_data import *

import warnings
import pickle

# global seed function for reproducibility 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.banchmark = False

# Main routine to train the neural net
def main(hparams, verbose = True):

    hparams.n_epochs = 400

    # Ignore unnecessary warnings
    warnings.filterwarnings("ignore")

    # Load data and create dataset
    if verbose: print('\n--- Creating dataset ---\n')
    time_dataset = time.time()
    dataset = create_dataset(hparams)
    print("Dataset created. Time elapsed:", datetime.timedelta(seconds=time.time()-time_dataset))
    node_features = dataset[0].x.shape[1]

    # Split dataset among training, validation and testing datasets
    train_loader, valid_loader, test_loader = split_datasets(dataset)

    # Size of the output of the GNN 
    dim_out = 2 * hparams.pred_params # in our case is 2 as mean and variance

    # Initialize model
    model = GNN(node_features=node_features,
                n_layers=hparams.n_layers,
                hidden_channels=hparams.hidden_channels,
                linkradius=hparams.r_link,
                dim_out=dim_out)
    
    model.to(device)

    # Print the memory (in GB) being used now:
    process = psutil.Process()
    print(f"Memory being used (GB): {process.memory_info().rss/1.e9:.3f}")

    # Train the net
    if verbose: print("\n--- Training ---\n")
    train_losses, valid_losses, chi2s = training_routine(model, train_loader, valid_loader, test_loader, hparams, verbose)

    # Test the model
    testing = True
    if testing:
        if verbose: print('\n--- Testing mode on ---\n')
        test_loss, rel_err, test_chi2 = test(test_loader, model, hparams)
        if verbose: print("Test Loss: {:.2e}, Relative error: {:.2e}, chi: {:.2e}".format(test_loss, rel_err, test_chi2))
    else :
        print('\n--- Validation mode on ---\n')
        print("Test Loss: {:.2e}, Relative error: {:.2e}, chi: {:.2e}".format(valid_losses, rel_err, chi2s))

    # Plot loss trends
    plot_losses(train_losses, valid_losses, hparams, display = False)

    # Plot true vs predicted params
    plot_out_true_scatter(hparams, "Om", display = False)
    if hparams.pred_params==2:
        plot_out_true_scatter(hparams, "Sig", display = False)

    return min(valid_losses), min(chi2s)

# --- MAIN ---#

if __name__ == "__main__":
    
    time_ini = time.time()

    set_seed(73)

    # Load hyperparameters
    fname = "best_hparams.pkl"
    with open(fname, 'rb') as file:
        best_hparams = pickle.load(file)

    main(best_hparams)

    # print hparams
    print('\nBest hyperparameters:')
    print('\tlearnig_rate: {}'.format(best_hparams.learning_rate))
    print('\tT_max: {}'.format(best_hparams.T_max))
    print('\tweight_decay: {}'.format(best_hparams.weight_decay))
    print('\tn_layers: {}'.format(best_hparams.n_layers))
    print('\thidden_channels: {}'.format(best_hparams.hidden_channels))

    print("Finished. Time elapsed:",datetime.timedelta(seconds=time.time()-time_ini))
