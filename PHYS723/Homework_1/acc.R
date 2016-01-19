library("rjson")
library("ggplot2")
json_data <- fromJSON(file="acc.json")
acc <- json_data$Accelerator
acc_df <- do.call("rbind", acc)
acc_df <- data.frame(acc_df)
acc_df[, c(2:3)] <- sapply(acc_df[, c(2:3)], as.numeric)
acc_df[, c(1)] <- sapply(acc_df[, c(1)], as.factor)
x_axis <- acc_df[,"Year"]
y_axis <- acc_df[,"Energy_MeV"]

for(i in 1:nrow(acc_df)){
    acc_df[i,"logE"] <- log(acc_df[i,"Energy_MeV"])
}

ggplot(acc_df, aes(x=Year, y=logE, color=Type)) + geom_point(shape=1) + scale_colour_hue(l=50)
ggsave("acc_R.pdf")