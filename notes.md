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
