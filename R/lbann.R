#' A LBA-NN Model Function
#'
#' This function allows you to build up a LBA-NN model with customized hyperparameters.
#' @importFrom magrittr %>%
#' @import ggplot2
#' @import keras
#' @import tensorflow
#' @param formula an object of class "formula" or one that can be coerced to that class with a response variable.
#' @param data a data frame to interpret the variables named in the formula.
#' @param num.layer the number of neurons in the hidden layer. The default value is NULL.
#' @param activation.1 the activation function in the hidden layer, one of “linear” or “relu”. If the value is missing, “linear” activation is used.
#' @param activation.2 the activation function in the output layer, one of “linear”, “relu” or “softmax”. If the value is missing, “linear” activation is used.
#' @param loss.function objective function to represent the error. If the value is missing, “mse” is used.
#' @param epochs the number of epochs or iterations to train the model. The default value is 10. See also “fit.keras.engine.training.Model” in keras.
#' @param val_split.ratio the ratio between the number of observations in the training set and validation set. The default value is 0.2.
#' @param lr learning rate. The default value is 0.01
#' @param K the number of predefined clusters. The value is recommended to be the same as in LBA
#' @param seed the seed for LBANN-Kmeans approach to achieve reproducible results
#' @keywords LBA-NN model
#' @return a list of the model, input matrix, output matrix, weight matrices, importance table and importance plot
#' @examples
#' lbann()
#' @export

lbann <- function(formula, data,
                  interaction = F,
                  num.layer, activation.1= "linear", activation.2 = "linear",
                  loss.fun = "mse", epochs = 10, val_split.ratio = 0.2,
                  lr = 0.01,
                  K = NA, seed = 1) {
  if (!inherits(formula,"formula")) {
    stop("Invalid type of argument: formula!")
  }
  if (!is.matrix(data) & !is.data.frame(data)) {
    stop("Invalid type of argument: data!")
  }
  if (!is.logical(interaction)) {
    stop("Invalid type of argument: interaction!")
  }
  if (!is.numeric(num.layer)) {
    stop("Invalid type of argument: num.layer! num.layer can only accept a numerical value!")
  }
  if (!activation.1 %in% c("linear", "relu", "softmax")) {
    stop("Invalid value of activation.1!")
  }
  if (!activation.2 %in% c("linear", "relu", "softmax")) {
    stop("Invalid value of activation.2!")
  }
  if (!loss.fun == "mse") {
    stop("Invalid value of loss.fun!")
  }
  if (!is.numeric(epochs)) {
    stop("Invalid type of argument: epochs! epochs can only accept a numerical value!")
  }
  if (!is.numeric(val_split.ratio)) {
    stop("Invalid type of argument: val_split.ratio! val_split.ratio can only accept a numerical value!")
  }
  if (!is.numeric(lr)) {
    stop("Invalid type of argument: lr! lr can only accept a numerical value!")
  }
  if (!is.numeric(K) & !is.na(K)) {
    stop("Invalid type of argument: K! K can only accept a numerical value!")
  }
  if (!is.numeric(seed)) {
    stop("Invalid type of argument: seed! seed can only accept a numerical value!")
  }

  aux.form <- strsplit(as.character(formula), split = "~")
  response.var <- aux.form[[2]]
  if (aux.form[3] == ".") {
    explanatory.var <- colnames(data)[-which(colnames(data) == response.var )]
  } else {
    explanatory.var <- unlist(strsplit(aux.form[[3]], ' \\+ '))
  }
  # Response variable and make the output matrix
  num.cat.resp <- length(unique(data[[response.var]]))
  name.cat.resp <- levels(as.factor(data[[response.var]]))
  resp.matrix <- keras::to_categorical(as.numeric(as.factor(data[[response.var]]))-1, num.cat.resp)
  colnames(resp.matrix) <- name.cat.resp
  # Explanatory variables and make the input matrix
  if (interaction == F) {
    num.cat.exp <- NULL
    exp.matrix <- NULL
    name.cat.exp <- NULL
    for (i in 1:length(explanatory.var)) {
      num.cat.exp[i] <- length(unique(data[[explanatory.var[i]]]))
      exp.matrix <- cbind(exp.matrix, keras::to_categorical(as.numeric(as.factor(data[[explanatory.var[i]]]))-1, num.cat.exp[i]))
      name.cat.exp <- c(name.cat.exp, levels(as.factor(data[[explanatory.var[i]]])))
    }
    colnames(exp.matrix) <- name.cat.exp
  } else {
    interaction.var <- data[[explanatory.var[1]]]
    for (i in 2:length(explanatory.var)) {
      interaction.var <- paste0(interaction.var, "+", data[[explanatory.var[i]]])
    }
    #interaction.var <- paste0(interaction.var, data[[explanatory.var[-1]]])
    num.cat.exp <- length(unique(interaction.var))
    exp.matrix <- keras::to_categorical(as.numeric(as.factor(interaction.var))-1, num.cat.exp)
    name.cat.exp <- levels(as.factor(interaction.var))
    colnames(exp.matrix) <- name.cat.exp
  }

  # define the layers
  # , bias_constraint = keras::constraint_maxnorm(0), kernel_constraint = keras::constraint_nonneg()
  input.layer <- keras::layer_input(shape=c(sum(num.cat.exp)))
  layers <- input.layer %>%
    keras::layer_dense(units = num.layer, activation = activation.1) %>%
    keras::layer_dense(units = length(name.cat.resp), activation = activation.2)
  # define the neural network model
  lba.nn.model <- keras::keras_model(inputs = input.layer, outputs = layers)
  lba.nn.model %>% keras::compile(
    keras::optimizer_rmsprop(lr),
    loss = loss.fun,
    metrics = c("accuracy")
    #tf$keras$metrics$AUC())#,
    # tf$keras$metrics$Precision())#tf$keras$metrics$AUC()
  )
  lba.nn.model %>% keras::fit(
    exp.matrix,
    resp.matrix,
    epochs = epochs,
    verbose = 1,
    validation_split = val_split.ratio
  )

  # weight matrix 1
  weight.matrix.1 <- keras::get_weights(lba.nn.model)[[1]]
  rownames(weight.matrix.1) <- name.cat.exp

  # weight matrix 2
  weight.matrix.2 <- keras::get_weights(lba.nn.model)[[3]]
  colnames(weight.matrix.2) <- name.cat.resp

  importance_data <- expand.grid(name.cat.exp, name.cat.resp)
  colnames(importance_data) <- c("explanatory", "response")
  importance_data['value'] <- NA

  for (i in 1:nrow(importance_data)) {
    importance_data[i,3] <- sum(weight.matrix.1[importance_data[i,1],] *
                                  weight.matrix.2[,importance_data[i,2]])
  }

  importance.plot <- ggplot2::ggplot(importance_data, ggplot2::aes(x = explanatory, y = value, fill=value)) +
    ggplot2::geom_bar(stat = "identity") +
    ggplot2::geom_hline(yintercept = 0) +
    ggplot2::scale_fill_viridis_c() +
    ggplot2::facet_wrap(~ response) +
    ggplot2::theme_bw() +
    ggplot2::labs(x = "", y = "importance") +
    ggplot2::theme(axis.text.x = ggplot2::element_text(angle = 90, vjust = 0.5, hjust=1),
                   legend.position = "none")

  summary.data <- matrix(NA, nrow = length(name.cat.resp), ncol = length(name.cat.exp))
  colnames(summary.data) <- name.cat.exp
  rownames(summary.data) <- name.cat.resp

  for (i in 1:sum(num.cat.exp)) {
    for (j in 1:length(name.cat.resp)){
      summary.data[j,i] <- sum(weight.matrix.1[i,] * weight.matrix.2[,j])
    }
  }

  ### LBANN-Kmeans Approach
  if (!is.na(K)) {
    set.seed(seed)
    pc <- prcomp(summary.data, center = T, scale. = T)
    data <- data.frame(obsnames=row.names(pc$x), pc$x,cluster = kmeans(summary.data, centers = K)$cluster)
    biplot <- ggplot(data, aes_string(x="PC1", y="PC2")) +
      geom_point(aes( color= as.factor(sapply(1:length(cluster), function(x) {paste("Cluster", cluster[x])})) ), alpha = 0.8) +
      geom_text(hjust=-0.3, vjust=0, size = 3, aes(label=obsnames,
                                                   color= as.factor(sapply(1:length(cluster), function(x) {paste("Cluster", cluster[x])}))) )+
      scale_color_discrete(name = "Clusters", labels = sapply(1:max(data$cluster), function(x) {paste("Cluster", x)})) +
      guides(color = guide_legend(title="Clusters") )
    biplot <- biplot + geom_hline(aes(yintercept=0), size=.2) + geom_vline(aes(xintercept = 0), size=.2)
    datapc <- data.frame(varnames=rownames(pc$rotation), pc$rotation)
    mult <- min(
      (max(data[,"PC2"]) - min(data[,"PC2"])/(max(datapc[,"PC2"])-min(datapc[,"PC2"]))),
      (max(data[,"PC1"]) - min(data[,"PC1"])/(max(datapc[,"PC1"])-min(datapc[,"PC1"])))
    )
    datapc <- transform(datapc,
                        v1 = .7 * mult * (get("PC1")),
                        v2 = .7 * mult * (get("PC2"))
    )
    biplot <- biplot + coord_equal() + geom_text(data=datapc, aes(x=v1, y=v2, label=varnames), size = 3, vjust=-0.8, alpha = 0.8, color = "black")#F90563
    biplot <- biplot + geom_point(data = datapc, aes(x = v1, y = v2), alpha = 0.8, shape = 25, color = "black", fill = "black")
    biplot <- biplot + theme_bw()
  }


  # return results
  if (is.na(K)) {
    structure(list(model = lba.nn.model,
                   input.matrix = exp.matrix,
                   output.matrix = resp.matrix,
                   weight.matrix.1 = weight.matrix.1,
                   weight.matrix.2 = weight.matrix.2,
                   importance = summary.data,
                   importance.plot = importance.plot,
                   importance.plot.data = importance_data), class = "lbann")
  } else {
    structure(list(model = lba.nn.model,
                   input.matrix = exp.matrix,
                   output.matrix = resp.matrix,
                   weight.matrix.1 = weight.matrix.1,
                   weight.matrix.2 = weight.matrix.2,
                   importance = summary.data,
                   importance.plot = importance.plot,
                   importance.plot.data = importance_data,
                   biplot = biplot), class = "lbann")
  }
}

