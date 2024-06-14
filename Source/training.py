# 
# Routines for training and testing the GNNs
# 

from Source.constants import *

# Training step
def train(loader, model, hparams, optimizer, scheduler):

    model.train()

    loss_tot = 0

    # Iterate in batches over the training dataset
    for data in loader:  

        data.to(device)
        optimizer.zero_grad()  # Clear gradients.
        out = model(data)  # Perform a single forward pass

        # Perform likelihood-free inference to predict also the standard deviation
        # Take mean and standard deviation of the output
        y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]

        # Compute loss as sum of two terms for likelihood-free inference
        loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
        loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1) , axis=0)
        loss = torch.log(loss_mse) + torch.log(loss_lfi)
        
        # Derive gradients
        loss.backward()  

        # Update parameters based on gradients
        optimizer.step()  
        scheduler.step() 
        loss_tot += loss.item()

    return loss_tot/len(loader)


# Testing/validation step
def test(loader, model, hparams):

    model.eval()

    trueparams = np.zeros((1, hparams.pred_params))
    outparams = np.zeros((1, hparams.pred_params))
    outerrparams = np.zeros((1, hparams.pred_params))

    errs = []
    chi2s = []
    loss_tot = 0

    # Iterate in batches over the training/test dat
    # aset --> can be optimized for GPU
    for data in loader:  
        with torch.no_grad():

            data.to(device)
            out = model(data)  # prediction of the model

            # perform likelihood-free inference to predict also the standard deviation
            # Take mean and standard deviation of the output
            y_out, err_out = out[:,:hparams.pred_params], out[:,hparams.pred_params:2*hparams.pred_params]

            # Compute loss as sum of two terms for likelihood-free inference
            loss_mse = torch.mean(torch.sum((y_out - data.y)**2., axis=1) , axis=0)
            loss_lfi = torch.mean(torch.sum(((y_out - data.y)**2. - err_out**2.)**2., axis=1), axis=0)
            loss = torch.log(loss_mse) + torch.log(loss_lfi)
            
            # absolute error
            err = (y_out - data.y)  / data.y  #/data.y --> CHECK
            errs.append(np.abs(err.detach().cpu().numpy()).mean() )

            # chi2
            chi2 = (y_out - data.y)**2 / err_out**2
            chi2s.append((chi2.detach().cpu().numpy()).mean())
            #chi2 = chi2s[chi2s < 1.e4].mean()
            
            loss_tot += loss.item()

            # Append true values and predictions
            trueparams = np.append(trueparams, data.y.detach().cpu().numpy(), 0)
            outparams = np.append(outparams, y_out.detach().cpu().numpy(), 0)
            outerrparams = np.append(outerrparams, err_out.detach().cpu().numpy(), 0)

    # Save true values and predictions (for plotting)
    np.save("Outputs/true_values.npy", trueparams)
    np.save("Outputs/predicted_values.npy", outparams)
    np.save("Outputs/errors_predicted.npy", outerrparams)

    return loss_tot/len(loader), np.array(errs).mean(axis=0), np.array(chi2s).mean()


# Training procedure
def training_routine(model, train_loader, valid_loader, test_loader, hparams, verbose=True):

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams.learning_rate, weight_decay=hparams.weight_decay)
    # ----------- LR GRAVEYARD ------------
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=hparams.learning_rate, max_lr=1.e-3, cycle_momentum=False)
    # gamma = hparams.gamma_lr
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = hparams.T_max, eta_min = 0, last_epoch = -1)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     'min', 
    #     threshold=0.05, 
    #     threshold_mode = 'abs', 
    #     verbose=True)

    train_losses, valid_losses, test_losses, chi2s = [], [], [], []
    valid_loss_min, err_min = 1000., 1000.
    chi2_min = 1e6

    # Training loop
    for epoch in range(1, hparams.n_epochs+1):
        train_loss = train(train_loader, model, hparams, optimizer, scheduler)

        valid_loss, err, chi2 = test(valid_loader, model, hparams)
        test_loss, err, _ = test(test_loader, model, hparams)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        # test_losses.append(test_loss)
        chi2s.append(chi2)

        # Save model if it has improved --> CHECK
        if valid_loss <= valid_loss_min:
            if verbose: print("Validation loss decreased ({:.2e} --> {:.2e}).  Saving model ...".format(valid_loss_min,valid_loss)) # PERCHE' QUI METTE LA TEST LOSS??
            torch.save(model.state_dict(), "Models/best_model_from_training")
            valid_loss_min = valid_loss
            err_min = err

        #if chi2 <= chi2_min :
        #   if verbose: print("Chi2 from validation decreased")
        #   torch.save(model.state_dict(), "Models/best_model_from_training")
        #   chi2_min = chi2

        if verbose: print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.2e}, Validation Loss: {valid_loss:.2e}, Error: {err:.2e}')

    return train_losses, valid_losses, chi2s
