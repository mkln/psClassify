# a (very) simple classification algorithm that estimates the
# probability that a name belongs to a person (and not to another entity
# eg. company or university)


library(glmnet)
library(ROCR)
library(plyr)

rm(list = ls())
setwd("/home/desktop/patstat_data/all_code/psClassify")

# read data from python
all_data <- read.table("r_data/r_input.csv", sep="\t", header=TRUE)
all_data_classify <- subset(all_data[all_data$certain_not_person == 0,], select= -c(name))
all_data$pr_is_person <- NA
all_data$is_person_hat <- NA

labeled_data <- all_data_classify[!is.na(all_data_classify$is_person),]
yt <- labeled_data[,"is_person"]
y_weight <- labeled_data[, "patent_ct"]

model_formula <- is_person ~ country + patent_ct +
                        applicant_seq + inventor_seq + 
                        word_count + lots_of_patents +
                        patent_ct + name_abbreviated +
                        avg_word_len + string_len + 
                        only_letters + has_legal_out +
                        has_legal_in + maybe_foreign_legal +
                        has_first_name + 
                        interaction(country, word_count) +
                        interaction(country, has_legal_out)
                        
X <- model.matrix(model_formula, all_data_classify)

# fit a L1-penalized logistic, weight by patent count
model <- cv.glmnet(X, yt, family = "binomial", weights = y_weight)

xb <- predict(model, X, s=model$lambda.min)
p_hat <- 1/(1+exp(-xb))

# precision and recall
precrec <- performance(prediction(p_hat, yt), "prec", "rec")

recall <- as.numeric(precrec@x.values[[1]])
recall[is.na(recall)] <- 0

precision <- as.numeric(precrec@y.values[[1]])
precision[is.na(precision)] <- 0

cutoffs <- as.numeric(precrec@alpha.values[[1]])

res_matrix <- as.matrix(cbind(precision, recall, cutoffs))
subset_recall1 <- res_matrix[res_matrix[,"recall"] == 1,]
max_precision <- max(subset_recall1[,"precision"])
print(paste("best precision for top recall =",max_precision))

# in this case, 'best' = 'max precision given recall=1'
# (in-sample)
best_cutoff <- subset_recall1[which(subset_recall1[,"precision"] == max_precision),"cutoffs"]

# is_person have been labeled
# certain_not_person should be filtered
the_rest <- all_data_classify
the_rest$is_person <- 0

# new sample
X_new <- model.matrix(model_formula, the_rest)

new_xb <- predict(model, as.matrix(X_new), s=model$lambda.min)
new_p_hat <- 1/(1+exp(-new_xb))
new_y <- rep(0, length(new_p_hat))
new_y[new_p_hat>best_cutoff] <- 1

all_data[all_data$certain_not_person == 0,"pr_is_person"] <- new_p_hat
all_data[all_data$certain_not_person == 0,"is_person_hat"] <- new_y

write.table(all_data, "r_out/r_output_all.csv", sep="\t", row.names=FALSE)

data_for_excel <- subset(all_data[all_data$certain_not_person == 0,], select = -c(patstat_id, applicant_seq, inventor_seq))
write.table(data_for_excel, "r_out/r_out_excel.csv", sep="\t", row.names=FALSE)

