# Models Performance on The Test Set

First seeing how much does the training data size changes the members disagreements and their mean error. And to check weather seed effects this measurements and models behaviour or not. I tested the model with 3 different seeds. And checked how each of the variants Pearson Correlation changes only when we changed the seed.
> Check [Convergence](C:\Users\Administrator\Desktop\code\comp_design_fw\results\convergence.csv)
This file denotes the each training size and the 3 seeds used each ones produced mean Pearson Correlation of Mean Error vs Member Disagreement.

Now for each size class of training data, we try different seeds to see how much does the correlations vary. We tried 3 seeds and their variation in std is plotted as the size of each point. And their x axis points to the training data that was used to train that model. And we see a downward trend in mean pearson coefficient with increasing the training data size... Although its not consistant.