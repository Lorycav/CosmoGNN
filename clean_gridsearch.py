import optuna
from optuna.visualization import plot_optimization_history, plot_contour, plot_param_importances    
from clean_hyperparameters import hparams
from clean_main import *

import pickle
import os

# Function to delete study if exists
def delete_study_if_exists(study_name: str, storage_url: str):
    try:
        # Attempt to load the study
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        
        # If the study is successfully loaded, delete it
        optuna.delete_study(study_name=study_name, storage=storage_url)
        print(f"Study '{study_name}' has been deleted.")

    except KeyError:
        # If the study does not exist, KeyError is raised
        print(f"Study '{study_name}' does not exist, nothing to delete.")

# Objective function to minimize
def objective(trial):

    # Hyperparameters to optimize --> there could be others
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    T_max =  trial.suggest_int("T_max", 5, 50 , log=False)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden_channels = trial.suggest_categorical("hidden_channels", [8, 16, 32, 64])

    # Some verbose
    print('\nTrial number: {}'.format(trial.number))
    print('\tlearning_rate: {}'.format(learning_rate))
    print('\tT_max: {}'.format(T_max))
    print('\tweight_decay: {}'.format(weight_decay))
    print('\tn_layers:  {}'.format(n_layers))
    print('\thidden_channels:  {}'.format(hidden_channels))

    # Hyperparameters to be optimized
    hparams.learning_rate = learning_rate
    hparams.T_max = T_max
    hparams.weight_decay = weight_decay
    hparams.n_layers = n_layers
    hparams.hidden_channels = hidden_channels

    # Run main routine
    min_val_loss, chi2s = main(hparams, verbose=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return min_val_loss, chi2s

# --- MAIN ---#
if __name__ == "__main__":

    time_ini = time.time()

    # Optuna parameters
    storage = "sqlite:///"+os.getcwd()+"/optuna_QUIJOTE"
    study_name = "clean_gnn"
    n_trials   = 40

    # Delete study if already present
    print('Deleting study named', study_name)
    delete_study_if_exists(study_name=study_name, storage_url=storage)

    # Define sampler and start optimization
    sampler = optuna.samplers.TPESampler(n_startup_trials=10)
    study = optuna.create_study(directions=['minimize', 'minimize'], study_name=study_name, sampler=sampler, storage=storage, load_if_exists=False)
    study.optimize(objective, n_trials, gc_after_trial=True)

    # Print info for best trial 
    trial = min(study.best_trials, key=lambda t: t.values[0]) # show best trial ------> t.values[0] for val loss,  t.values[1] for chi 
    print("Best trial:")
    print("  Validation loss Value: ", trial.values[0])
    print("  Chi2 Value: ", trial.values[1])
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    hparams.learning_rate = trial.params["learning_rate"]
    hparams.T_max = trial.params['T_max']
    hparams.weight_decay = trial.params["weight_decay"]
    hparams.n_layers = trial.params["n_layers"]
    hparams.hidden_channels = trial.params["hidden_channels"]

    # Save best model 
    best_hpars_file = 'best_hparams.pkl'
    with open(best_hpars_file, 'wb') as file:
        pickle.dump(hparams, file)

    # Visualization of optimization results (with optuna functions)
    fig = plot_optimization_history(study, target=lambda t: t.values[0])
    fig.write_image("Plots/optuna_optimization_history.png", width = 1200, height = 800)
    
    fig = plot_contour(study, target=lambda t: t.values[0])
    fig.write_image("Plots/optuna_contour.png", width = 1200, height = 800)

    fig = plot_param_importances(study, target=lambda t: t.values[0])
    fig.write_image("Plots/optuna_param_importances.png", width = 1200, height = 800)

    print("END --- time elapsed:", datetime.timedelta(seconds=time.time()-time_ini))
