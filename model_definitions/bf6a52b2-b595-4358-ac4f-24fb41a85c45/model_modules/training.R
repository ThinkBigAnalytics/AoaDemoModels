library("gbm")

load_data <- function() {
    N <- 1000
    X1 <- runif(N)
    X2 <- 2*runif(N)
    X3 <- factor(sample(letters[1:4],N,replace=T))
    X4 <- ordered(sample(letters[1:6],N,replace=T))
    X5 <- factor(sample(letters[1:3],N,replace=T))
    X6 <- 3*runif(N)
    mu <- c(-1,0,1,2)[as.numeric(X3)]

    SNR <- 10 # signal-to-noise ratio
    Y <- X1**1.5 + 2 * (X2**.5) + mu
    sigma <- sqrt(var(Y)/SNR)
    Y <- Y + rnorm(N,0,sigma)

    # create a bunch of missing values
    X1[sample(1:N,size=100)] <- NA
    X3[sample(1:N,size=300)] <- NA

    # random weights if you want to experiment with them
    # w <- rexp(N)
    # w <- N*w/sum(w)
    w <- rep(1,N)

    data <- data.frame(Y=Y,X1=X1,X2=X2,X3=X3,X4=X4,X5=X5,X6=X6)
    data
}


train <- function(data_conf, model_conf, ...) {

    data <- load_data()
    hyperparams <- model_conf[["hyperParameters"]]

    model <- gbm(Y~X1+X2+X3+X4+X5+X6,
            data=data,
            var.monotone=c(0,0,0,0,0,0),
            distribution="gaussian",
            n.trees=hyperparams$n.trees,
            shrinkage=hyperparams$shrinkage,
            interaction.depth=hyperparams$interaction.depth,
            bag.fraction=hyperparams$bag.fraction,
            train.fraction=hyperparams$train.fraction,
            n.minobsinnode=hyperparams$n.minobsinnode,
            keep.data=TRUE,
            cv.folds=hyperparams$cv.folds,
            verbose = FALSE)

    best.iter <- gbm.perf(model, plot.it=FALSE, method="cv")

    # clean the model
    light <- model
    light$trees <- light$trees[best.iter]
    light$data <- list()

    saveRDS(light, "models/model.rds")
}
