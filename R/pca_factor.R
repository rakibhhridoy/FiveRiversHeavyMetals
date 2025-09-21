winter = read.csv("/home/RHHLab/Research/Five_Rivers/Python/winter.csv")
rainy =  read.csv("/home/RHHLab/Research/Five_Rivers/Python/rainy.csv")
winter <- winter[,-1]

winter
fc <- factanal(winter, factors = 1, rotation = "varimax")
fc

fc$loadings

write.csv(fc$loadings, "Factor_loadings_Winter.csv")
fc <- factanal(rainy, factors = 1, rotation = "varimax")
fc

write.csv(fc$loadings, "Factor_loadings_rainy.csv")
require(psych)

pc <- pca(winter, nfactors = 1, rotate = "varimax")
pc

pc$weights[1,]

require(factoextra)
require(FactoMineR)

sample2 <- PCA(data.frame(winter), graph = FALSE, ncp = 1,scale.unit = TRUE)
eig.val <- get_eigenvalue(sample2)
eig.val

sample2 <- PCA(data.frame(rainy), graph = FALSE, ncp = 1,scale.unit = TRUE)
eig.val <- get_eigenvalue(sample2)
eig.val

pc$weights

write.csv(pc$loadings, "PCA_loading_Winter.csv")
write.csv(pc$weights, "PCA_weight_Winter.csv")
pc <- pca(rainy, nfactors = 1, rotate = "varimax")
pc

pc$weights