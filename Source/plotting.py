# Script for plotting some statistics

import matplotlib.pyplot as plt
from Source.constants import *
from sklearn.metrics import r2_score
from matplotlib.offsetbox import AnchoredText
from matplotlib.ticker import MultipleLocator
from matplotlib.lines import Line2D
import matplotlib as mpl
# mpl.rcParams.update({'font.size': 12})

# Plot loss trends
def plot_losses(train_losses, valid_losses, hparams, display):

    epochs = hparams.n_epochs

    fig_losses, ax = plt.subplots(figsize=(12,8))
    ax.grid(alpha = 0.4)

    # ax.plot(range(epochs), np.exp(train_losses), label = 'Training')
    # ax.plot(range(epochs), np.exp(valid_losses), label = 'Validation', linestyle = '-')
    ax.plot(range(epochs), train_losses, label = 'Training')
    ax.plot(range(epochs), valid_losses, label = 'Validation', linestyle = '-')

    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    # ax.set_ylim(0,15)
    # ax.set_yscale('log')

    ax.set_title('Train and validation losses')

    # plt.title(f"Test loss: {test_loss:.2e}, Minimum relative error: {err_min:.2e}")
    if display: plt.show()
    fig_losses.savefig("Plots/losses.png", bbox_inches='tight', dpi=400)
    plt.close(fig_losses)

# Remove normalization of cosmo parameters
def denormalize(trues, outputs, errors, minpar, maxpar):

    trues = minpar + trues*(maxpar - minpar)
    outputs = minpar + outputs*(maxpar - minpar)
    errors = errors*(maxpar - minpar)
    return trues, outputs, errors


# Scatter plot of true vs predicted cosmological parameter
def plot_out_true_scatter(hparams, cosmoparam, display):

    figscat, axscat = plt.subplots(figsize=(6,5))
    # axscat.grid(linestyle='--', alpha=0.3)
    axscat.grid(alpha=0.3)

    # Load true values and predicted means and standard deviations
    trues = np.load("Outputs/true_values.npy")
    outputs = np.load("Outputs/predicted_values.npy")
    errors = np.load("Outputs/errors_predicted.npy")

    # There is a (0,0) initial point, fix it
    outputs = outputs[1:]
    trues = trues[1:]
    errors = errors[1:]

    # Choose cosmo param and denormalize
    if cosmoparam=="Om":
        minpar, maxpar = 0.1, 0.5
        outputs, trues, errors = outputs[:,0], trues[:,0], errors[:,0]
    elif cosmoparam=="Sig":
        minpar, maxpar = 0.6, 1.0
        outputs, trues, errors = outputs[:,1], trues[:,1], errors[:,1]
    trues, outputs, errors = denormalize(trues, outputs, errors, minpar, maxpar)

    # Compute the number of points lying within 1 or 2 sigma regions from their uncertainties
    cond_success_1sig, cond_success_2sig = np.abs(outputs-trues) <= np.abs(errors), np.abs(outputs-trues) <= 2.*np.abs(errors)
    tot_points = outputs.shape[0]
    successes1sig, successes2sig = outputs[cond_success_1sig].shape[0], outputs[cond_success_2sig].shape[0]

    # Compute the linear correlation coefficient
    r2 = r2_score(trues, outputs)
    # mean relative error
    err_rel = np.mean(np.abs((trues - outputs) / trues), axis=0)
    # chi^2
    chi2 = (outputs - trues)**2. / errors**2.
    chi2 = np.mean(chi2)
    #chi2 = chi2s[chi2s < 1.e4].mean()    # Remove some outliers which make explode the chi2

    # print("R^2 = {:.2f}, Relative error = {:.2e}, Chi2 = {:.2f}".format(r2, err_rel, chi2))
    print(f'R2: {r2:.3f}')
    print(f'relative error: {err_rel:.3f}')
    print(f"A fraction of succeses of {successes1sig/tot_points:.3f} at 1 sigma, {successes2sig/tot_points:.3f} at 2 sigmas")

    # Sort by true value
    indsort = trues.argsort()
    outputs, trues, errors = outputs[indsort], trues[indsort], errors[indsort]

    # Compute mean and std region within several bins --> CHECK
    # truebins, binsize = np.linspace(trues[0], trues[-1], num=10, retstep=True)
    # means, stds = [], []
    # for i, bin in enumerate(truebins[:-1]):
    #     cond = (trues>=bin) & (trues<bin+binsize)
    #     outbin = outputs[cond]
    #     if len(outbin)==0:
    #         outmean, outstd = np.nan, np.nan    # Avoid error message from some bins without points
    #     else:
    #         outmean, outstd = outbin.mean(), outbin.std()
    #     means.append(outmean); stds.append(outstd)
    # means, stds = np.array(means), np.array(stds)
    # print("Std in bins:",stds[~np.isnan(stds)].mean(),"Mean predicted uncertainty:", np.abs(errors.mean()))

    # Plot predictions vs true values
    truemin, truemax = trues.min(), trues.max()
    #truemin, truemax = minpar-0.05, maxpar+0.05
    #axscat.plot([truemin, truemax], [0., 0.], "r-")
    #axscat.errorbar(trues, outputs-trues, yerr=errors, color=col, marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    axscat.plot([truemin, truemax], [truemin, truemax], "r-")
    axscat.errorbar(trues, outputs, yerr=np.abs(errors), marker="o", ls="none", markersize=0.5, elinewidth=0.5, zorder=10)
    #axscat.fill_between(trues, outputs-np.abs(errors), outputs+np.abs(errors), alpha=0.6)

    # Legend
    # if cosmoparam=="Om":
    #     par = "\t"+r"$\Omega_m$"
    # elif cosmoparam=="Sig":
    #     par = "\t"+r"$\sigma_8$"
    # leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    leg = "$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.1f} %".format(100.*err_rel)+"\n"+"$\chi^2$={:.2f}".format(chi2)
    #leg = par+"\n"+"$R^2$={:.2f}".format(r2)+"\n"+"$\epsilon$={:.2e}".format(err_rel)
    at = AnchoredText(leg, frameon=True, loc="upper left")
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    axscat.add_artist(at)

    # Labels etc
    # axscat.set_xlim([truemin, truemax])
    # axscat.set_ylim([truemin, truemax])
    axscat.set_ylabel(r"Prediction")
    axscat.set_xlabel(r"Truth")

    axscat.grid(linestyle='--', alpha=0.8)

    # Title, indicating which are the training and testing suites
    # if hparams.only_positions:
    #     usefeatures = "Only positions"
    # else:
    #     usefeatures = r"Positions + $V_{\rm max}, M_*, R_*, Z_*$"
    # props = dict(boxstyle='round', facecolor='white')#, alpha=0.5)
    # axscat.text((minpar + maxpar)/2-0.02, minpar, usefeatures, color="k", bbox=props )

    axscat.set_title('True vs predicted $\\Omega_m$')
    if display: plt.show()
    figscat.savefig("Plots/true_vs_pred.png", bbox_inches='tight', dpi=300)
    plt.close(figscat)
