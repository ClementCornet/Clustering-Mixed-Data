library(kamila)
library("rjson")

k <- fromJSON(file = "k.json")$n_clusters

con_vars <- read.csv(file = "temp_continue.csv")
con_vars <- data.frame(scale(con_vars))

cat_vars_fac <- read.csv(file = "temp_cat.csv")

cat_vars_fac[] <- lapply(cat_vars_fac, factor)
cat_vars_dum <- dummyCodeFactorDf(cat_vars_fac)
cat_vars_dum <- data.frame(cat_vars_dum)


gms_res_hw <- gmsClust(con_vars, cat_vars_dum, nclust = k)

clusters <- gms_res_hw$results$cluster - 1

df <- cbind(
    con_vars,
    cat_vars_fac,
    cluster = clusters
)

write.csv(df, "temp_clustered.csv", row.names = FALSE)