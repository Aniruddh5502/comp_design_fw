> Research Logs

# Can I trust the process that made the Surrogate

## 2. The Question
Predicting and trying to find any way to tweak the architecture so that the error estimation can predict the amount of absolute error. And have a Pearson/Spearman correlation between Error Prediction VS Absolute Error.

## Arch Choice
First we tested some fany architectures to see if they hold up with the expectations. I tested Ensemble Surrogates, SNGP on multi output regression type problem where we use FEA data from a SDOF compliant mechanicsms Static Structural and Modal Analysis information. And from the results we have seen that there is minimal relation/correlation between the error prediction from member varaince in Ensemble and GP prediction from SNGP that doesn't strongly relate the error prediction with absolute error. Both of those test cases had $<0.5$ in both absolute Spearman and Pearson Correlation.

Now I am testing vanilla NN on this 1200+ datapoint training data from FEA simulations to train a Vanilla NN to predict its outputs, and then we shall see the error measurements after we test it with the test set and try to see the pattern in which the error varies.

Then we take another unseen portion of the dataset from the same experiment and compute the error on those datapoints. Then use that map of error behaviour of that model train a new model for only error prediction. Once this error prediction model is accurate enough that we can have a correlation $>0.5$ then we can say we have a good predictor which will be tested with the held out test set obviously.

So total dataset division will be training set, error map generation set, test set. I think these 3 sets will be enough. 


## Mappings

```text 

config/             # Configuration settings
data/               # All the datasets and Ansys Simulation files
plots/              # plots from experiments
scaler/             # Dataset Scaling data
scripts/            # All scripts includes Ansys, simulations etc
src/                # Constains core coders, scripts to run the simulation pipeline
tests/              # Ansys, Sim, Loop, etc tests
writings/           # writing the papers and its drafts, nots etc
.gitignore          # git settings
README.md           # This file.
requirements.txt    # Requirements or dependencies to run the experiments in pyhton

``` 

## Running

### Running Simulation Pipeline

### Model Training and Post Steps

### Conclusion