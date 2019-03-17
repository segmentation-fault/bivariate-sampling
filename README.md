# bivariate-sampling
Python implementation to generate samples from a bivariate continuous distribution

## Some theory
For a univariate distribution with CDF ![equation](https://latex.codecogs.com/gif.latex?F_X%28x%29%20%3D%20%5CPr%5C%7B%20X%20%3C%20x%20%5C%7D) , we can generate a sample from said distribution via the inverse sampling theorem; if the random variable U is drawn from a uniform distribution between 0 and 1, then a realization of the random variable X is:

![equation](https://latex.codecogs.com/gif.latex?X%20%3D%20F%5E%7B-1%7D%20%28U%29) ,

or, the inverse of the CDF of X calculated in a random outcome of the uniform distribution between 0 and 1.

For a bivariate distribution with CDF ![equation](https://latex.codecogs.com/gif.latex?F_%7BX%2CY%7D%28x%2Cy%29), we can approximate the generation of two samples by following those two steps:

- Generate a sample from X with the Inverse sampling theorem, using the marginal CDF ![equation](https://latex.codecogs.com/gif.latex?F_%7BX%7D%28x%29)
- Then, generate a sample of Y by using the conditional CDF of Y given that X=x, i.e. 

![equation](https://latex.codecogs.com/gif.latex?F_%7BY%20%5Clvert%20X%3Dx%7D%28y%29%20%3D%20%5Cfrac%7BF_%7BX%2CY%7D%28X%3Dx%2CY%29%7D%7BF_X%28X%3Dx%29%7D) .

In case the marginal is not readily available, one can calculate it as:

![equation](https://latex.codecogs.com/gif.latex?F_X%28x%29%20%3D%20%5Clim_%7By%20%5Crightarrow%20&plus;%20%5Cinfty%7D%20F_%7BX%2CY%7D%28x%2Cy%29) .

In practice this means calculating the Joint CDF at a very high value of y (in my code I used 1e5).

