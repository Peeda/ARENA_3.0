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
# Chapter 1: Interpretability
## Part 1: Transformers
* what's the input output dims, we batch but also do each sequence pos prediction in parallel
    * for each batch we do this attention calculation
* Token and positional embeddings (why position?)
* general architecture, blocks and the residual stream
    * layer norm before heads, before mlp also
    * gelu
* Attention is calculated as QK^T / sqrt(d_head), where Q,K are the results of mapping residual stream into d_model
* softmax to get the weights for our convex combination, take lin combos of V (the information passed by each tok)
* intuition of guys standing in line asking questions backwards
* causal masking

* greedy vs beam vs direct sampling vs top k vs top p
* logits vs log probs (log probs are unique right? it wasnt mentioned but i remember hearing that)
## Part 2: Intro to Mech Interp
### conceptual
* embed and unembed can only expression bigrams because it only looks at the past?
* each head places some data on the residual stream, in its own subspace
* we can decompose the output in terms of the contribution of heads
* skip trigrams and induction heads
    * make sure to mentally differentiate between the residual stream vector at a certain position (in d_model) and Q,K,V vectors (in d_head)
        * the QK, OV circuits are just in terms of the weights, not the inputs
        * maybe this is more day 2 though
    * look back at that diagram until it makes sense, in each step

### exercises, transformerlens
* transformer lens and the cache
* finding induction heads
    * we plot the attention patterns (the probs after softmax) and look for a prev token head
    * and also look for an induction head in a random repeated seq
    * we do this by plotting the attention pattern for each head in each layer and looking at it
* hooks are functions that intervene after an activation has been calculated
    * we can make a hook that checks each attention pattern for inductionness, then we can loop over the desired layers and visualize patterns for closer look
* implement logit attribution
    * review what precisely we're doing, how we look at just the correct token and why this gives s-1 type dims
* ablation
    * you zero out an activation, or maybe take the mean
    * compare how the loss looks to the baseline over the whole seq after to get some idea of how this head contributes
