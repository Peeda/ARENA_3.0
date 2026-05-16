# Chapter 0: Fundamentals
## Part 1: Ray Tracing
* Einops, gives rearrange, repeat, reduce
## Part 2: CNNS
* What is a module, parameter, sequential
* modules typically have __init__, forward
* what does a training loop look like, what are dataloaders
    * we got the train loop, and the validation step
* what are convolutions, why channels for images
* what is batch normalization
* train and eval mode
* resnet architecture at a high level, blockgroups consist of residual blocks
    * how does this optional conv basically behave like a skip layer
* what is feature extraction
## Part 3: Optimization
* batch vs sgd vs mini batch gd
* gradient descent, momentum, rmsprop, adam, adamw
* rmsprop and adam are adaptive, why does this mean we don't need to scale learning rate with batch size
* what is weight decay

* adam and adamw center first moments, why does this make early values large

* how does wandb look, what 3 methods are you basically calling
* what are sweeps
## Part 4: backprop
* if you were to push the gradients forward you collect all the gradients you need at L
    * you traverse the whole graph to just get a specific gradient propagated to the root
    * there's an asymmetry in push forward vs push back I think maybe because both are a special case of vector to vector fns?
* why do we unbroadcast at the end when doing the elementwise multiply back_fn
* maybe derive the chain rule for elementwise ops, why does the intuitive thing actually work
* make sure im comfortable with how this is more general than just implementing a neural network with hardcoded weights
* review the matrix multiplication gradient derivation, this type of usage of chain rule as a sum still feels mysterious
    * probably relates to feeding forward vs pushing back, do you apply a forwards derivative operator or a backwards one
* maybe look back over the tensor code? It feels like I'm kinda relying on type checking and there's some important wrapping stuff going unnoticed like for .T and stuff like that, want to get an understanding for how tensors are created on evaluating expressions
## Part 5: VAEs and GANs
* review the intuitions for kl, how it's the same as ce up to an additive factor
* how do aes relate to pca, become prone to overfitting, and how do vaes fix this
    * specifically how does the tradeoff between the two losses interpolate between extremes
* what's the trick that allows us to backprop through sampling
* it seems like both clumped/concentrated clustering is bad (ae) but also far apart clustering (which we avoid with kl right), understand both
    * oh wait maybe the clumping was just bc the ae scale was larger (-10 to 10) whereas the vae uses -2 to 2 which is more like N(0,1)
