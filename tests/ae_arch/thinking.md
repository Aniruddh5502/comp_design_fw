What I am thinking is first do PCA on the data input size(4) inputs.
Get the latent size where 100% of data variability can be constrained into and take that dimension. Then use that size for the bottleneck for a Auto encoders bottleneck. And then this trained Encoder can be considered as the latent representor. And the output of this Encoder part is the latent representation of the inputs.

In this Autoencoder Architecture the loss function will be the reconstruction cost of the inputs from the latent space.

Now we train another model to predict the output from the latent representations, here the loss is the distance from the output/target.
