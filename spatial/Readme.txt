These models have been calibrated removing the re3 and swir2 bands in one (8 bands), and removing re3, swir2 and blue bands in the other one (7 bands).
Therefore, they have to be applied to stacks of 8x16 time steps and 7x16 time steps.

Re3 and swir2 are highly correlated with re2 and swir1 respectively, and blue band tends to be noisy.
Crossvalidation results don't vary much with respect to the model that uses the 10 bands.
r2_hat: 0.46
r2_sd: 0.00
rrmse_hat: 0.27
rrmse_sd: 0.00
rmses_hat: 4.15
rmses_sd: 0.14
rmseu_hat: 7.06
rmseu_sd: 0.16
