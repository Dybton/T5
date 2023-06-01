install.packages("car")
install.packages("corrplot")

library(car)
library(tidyverse)
library(lme4)
library(lmerTest)


data <- read.csv('dev_df_with_results.csv')

glimpse(data) ## Make sure the data looks like it should

# Normalize the data
data <- data %>%
  mutate(
    num_args = scale(num_args),
    question_length = scale(question_length),
    nesting_level = scale(nesting_level),
    schema_total_complexity = scale(schema_total_complexity),
    schema_types_count = scale(schema_types_count),
    schema_fields_count = scale(schema_fields_count),
    schema_input_objects_count = scale(schema_input_objects_count),
    schema_relationships_count = scale(schema_relationships_count),
    schema_arguments_count = scale(schema_arguments_count),
    schema_length = scale(schema_length),
    query_length = scale(query_length)
  )

#Running the model

model <- glmer(
  semantic_match ~ schema_arguments_count + schema_relationships_count + schema_fields_count + num_args + schema_total_complexity + question_length + nesting_level + schema_length + schema_types_count + schema_input_objects_count +
    (1 | schemaId),
  data = data,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)


#We need to check for multicollinearity
vif_values <- vif(model)
print(vif_values)

correlation_matrix <- cor(data[c("schema_arguments_count", "schema_relationships_count", "schema_fields_count", "schema_total_complexity", "schema_length", "schema_types_count", "schema_input_objects_count")])
print(correlation_matrix)


library(corrplot)
corrplot(cor_matrix, method = "color")

# Standardize the variables
data$standardized_arguments <- scale(data$schema_arguments_count)
data$standardized_fields <- scale(data$schema_fields_count)
data$standardized_types <- scale(data$schema_types_count)
data$standardized_inputs <- scale(data$schema_input_objects_count)

# Create the composite measure
data$schema_composite_count <- (data$standardized_arguments + data$standardized_fields + data$standardized_types + data$standardized_inputs) / 4

glimpse(data) ## Make sure the data looks like it should

# Remove the individual variables from the data frame if desired
data <- subset(data, select = -c(standardized_arguments, standardized_fields, standardized_types, standardized_inputs))
data <- subset(data, select = -c(schema_arguments_count, schema_fields_count, schema_types_count, schema_input_objects_count))

glimpse(data) ## Make sure the data looks like it should


# Since they are so closely correlated we remove all but schema_total_complexity
correlation_matrix <- cor(data[c("num_args", "question_length", "nesting_level", "query_length", "schema_composite_count")])
print(correlation_matrix)

library(corrplot)
corrplot(correlation_matrix, method = "color")

model1 <- glmer(
  exact_match ~ num_args + schema_total_complexity + question_length + nesting_level + nesting_level + num_args + query_length +
    (1 | schemaId),
  data = data,
  family = binomial,
  control = glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

#We need to check for multicollinearity again
vif_values1 <- vif(model1)
print(vif_values1)

#See what the model is doing
summary(model1)




