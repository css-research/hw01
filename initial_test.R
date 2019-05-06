library(tidyverse)
library(keras)
library(tensorflow)

fashion_mnist = keras::dataset_fashion_mnist()

x_train = fashion_mnist$train$x
y_train = fashion_mnist$train$y

x_test = fashion_mnist$test$x
y_test = fashion_mnist$test$y

rm(fashion_mnist)

x_train = array_reshape(x_train, c(nrow(x_train), 28 * 28)) / 255
x_test = array_reshape(x_test, c(nrow(x_test), 28 * 28)) / 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

model = keras_model_sequential() 

model %>% 
  layer_dense(units = 512, activation = 'relu', input_shape = c(784)) %>% 
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 512, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

history = model %>% fit(
  x_train, y_train, 
  epochs = 200, batch_size = 512, 
  validation_split = 0.1666666666666667
)

plot(history)

model %>% evaluate(x_test, y_test)