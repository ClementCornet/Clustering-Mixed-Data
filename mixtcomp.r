library(RMixtComp)

library("rjson")

k <- fromJSON(file = "k.json")$n_clusters

data <- read.csv("heart_failure_long.csv")
data <- data[c(1, 10, 7, 9, 11)]
data[2] <- data[2] + 1
data$platelets <- data$platelets / 10
data[5] <- data[5] + 1

# "Gaussian" for numeric variable, "Multinomial" for character or factor variable and "Poisson" for integer variable # nolint

algo <- list(nbBurnInIter = 50,
             nbIter = 50,
             nbGibbsBurnInIter = 50,
             nbGibbsIter = 50,
             nInitPerClass = 20,
             nSemTry = 20,
             confidenceLevel = 0.95)

model <- list("age" = "Poisson",
              "sex" = "Multinomial",
              "serum_sodium" = "Poisson",
              "platelets" = "Gaussian",
              "smoking" = "Multinomial") # nolint

resLearn1 <- mixtCompLearn(data, model, algo,
                           nClass = k, nRun = 2, nCore = 1)

resPredict <- mixtCompPredict(data, model, algo,
                              resLearn1, nClass = k, nCore = 1)

clusters <- resPredict$variable$data$z_class$completed

write.csv(clusters, "mixtcomp_temp.csv", row.names = FALSE)