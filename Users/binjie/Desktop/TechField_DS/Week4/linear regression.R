N <- 100

x <- seq(1, 20, length.out = N)
y <- 5.524 + 3.035*x + rnorm(N, sd = 2.5)

d <- mean(x^2) - mean(x)^2

w0 <- (mean(x^2)*mean(y)-mean(x)*mean(x*y))/d
w1 <- (mean(x*y)-mean(x)*mean(y))/d

y_hat <- w0 + w1*x

plot(x,y)
lines(x,y_hat)
#plot(x,y_hat)

R2 <- 1 - sum((y-y_hat)^2)/sum((y-mean(y))^2)

cat("Training R-square:", round(R2, digits = 3))

    