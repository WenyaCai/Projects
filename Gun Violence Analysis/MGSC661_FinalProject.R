#libraries
library(tidyverse)
library(lubridate)
library(ggplot2)
library(rcartocolor)
library(maps)
library(scales)
library(knitr)
library(tidyr)
library(randomForest)
library(gbm)
library(rms)
library(caret)
library(reshape2)
library(corrr)
library(ggcorrplot)
library(usmap)
library(ggpubr)

d = read.csv("Gun violence.csv")
# drop unnecessary 
drop=c("incident_id","address","incident_url","source_url","incident_url_fields_missing","congressional_district","gun_stolen","notes","participant_name","participant_relationship","sources","state_house_district","state_senate_district")
d <- d[,!(names(d) %in% drop)]

#add columns: month, year, injuries + death, number of suspect
d$date <- ymd(d$date)
d$year <- year(d$date)
d$month <- month(d$date)
d$victims <- d$n_injured+d$n_killed

incidents_by_year <- d %>%
  group_by(year) %>%
  summarise(total_incidents = n())
incidents_by_year$year <- as.factor(incidents_by_year$year) 

# Preparing the data for total incidents by weekday
d$weekday <- wday(d$date, label = TRUE)
incidents_by_weekday <- d %>%
  group_by(weekday) %>%
  summarise(total_incidents = n())
incidents_by_weekday$weekday <- as.factor(incidents_by_weekday$weekday ) 

# Plot 1: Total Incidents by Year
ggplot(incidents_by_year, aes(x = year, y = total_incidents, fill = year)) + geom_bar(stat = "identity", show.legend = FALSE) +
  scale_fill_manual(values = c("2013"="#D2FBD4", "2014" = "#9CD5BE", "2015" = "#6CAFA9", "2016" = "#458892", "2017" = "#266377", "2018"="#123F5A")) + labs(title = "Total Incidents by Year", x = "Year", y = "Total Incidents") + theme_minimal()

# Plot 2: Total Incidents by Weekday
ggplot(incidents_by_weekday, aes(x = weekday, y = total_incidents, fill = weekday)) + geom_bar(stat = "identity", show.legend = FALSE) + scale_fill_manual(values = c("#F9DDDA", "#F2B9C4", "#E597B9", "#CE78B3", "#AD5FAD", "#834BA0" ,"#573B88")) +
  labs(title = "Total Incidents by Weekday", x = "Weekday", y = "Total Incidents") + theme_minimal()

incident_counts_state <- aggregate(. ~ state, data = d, FUN = length)
plot_usmap(regions = "states", data = incident_counts_state, values = "victims") + scale_fill_carto_c(name = "Incident Count", palette = "SunsetDark", direction = 1) +
  labs(title = "Heatmap of Shooting Incidents in the US",
       subtitle = "Count of Incidents by State") +
  theme_minimal() +
  theme(legend.position = "right",
        axis.title.x = element_blank(),
        axis.title.y = element_blank())

top_5_dangerous_states <- incident_counts_state %>% select("state","victims") %>%
  arrange(desc(victims)) %>%
  slice_head(n = 5)
knitr::kable(top_5_dangerous_states,"html")

#dataset only include partial 2013 and 2018, so we only look at 2014-2017
# Filter data for years 2014 to 2017
plot_data <- d %>% filter(date >= as.Date("2014-01-01") & date <= as.Date("2017-12-31"))

# Create a new column for non-year-specific date (month-day)
plot_data$month_day <- format(plot_data$date, "%m-%d")

# Group by year and month-day, and count the incidents
incident_counts <- plot_data %>%
  group_by(year = year(date), month_day) %>%
  summarise(incidents = n()) %>%
  ungroup()

# Spread the counts into a wide format with one column per year
incident_counts_wide <- spread(incident_counts, year, incidents)

# Replace NA with zeros for missing values (days without incidents)
incident_counts_wide[is.na(incident_counts_wide)] <- 0

# Gather the data for plotting
incident_counts_long <- gather(incident_counts_wide, year, incidents, -month_day)

# Convert the year to a factor for discrete coloring
incident_counts_long$year <- as.factor(incident_counts_long$year)


# Plot with specified breaks and custom colors for each year
ggplot(incident_counts_long, aes(x = as.Date(paste0("2014-", month_day)), y = incidents, group = year, color = year)) +
  geom_line() +
  scale_x_date(
    breaks = as.Date(paste0("2014-", sprintf("%02d-01", 1:12))), # First day of each month for the year 2014
    labels = function(x) format(x, "%m-%d") # Custom labels for the x-axis
  ) +
  scale_y_continuous(labels = scales::comma) +
  scale_color_manual(
    values = c("2014" = "#648FFF", "2015" = "#785EF0", "2016" = "#DC267F", "2017" = "#FE6100")) +
  labs(
    title = "Total Gun Violence Incidents by Date (2014-2017)",
    x = "Date (MM-DD)",
    y = "Incident Count",
    color = "Year"
  ) +
  theme_minimal() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "right"
  ) +
  guides(color = guide_legend(override.aes = list(size = 4))) # Increase legend line size for visibility

# Sum incidents across all years for each date
incident_counts_sum <- incident_counts_wide %>%
  rowwise() %>%
  mutate(total_incidents = sum(c_across(`2014`:`2017`), na.rm = TRUE)) %>%
  ungroup()

# Find the top 3 most dangerous dates
top_3_dangerous_dates <- incident_counts_sum %>%
  arrange(desc(total_incidents)) %>%
  slice_head(n = 3)

# Print the top 3 most dangerous dates
print(top_3_dangerous_dates)

cleaned_text <- na.omit(d$location_description)

word_freqs <- table(tolower(cleaned_text))
word_df <- data.frame(Location = names(word_freqs), Frequency = as.integer(word_freqs))
word_df = word_df[-1,]
top_5_location <- word_df %>%
  arrange(desc(Frequency)) %>%
  slice_head(n = 5)
kable(top_5_location)

#Model
df = read.csv("Gun violence filtered.csv")

features = c("date","state","city_or_county","n_killed","n_injured","suspect_age_group","suspect_exact_age","suspect_gender","incident_characteristics","congressional_district","state_senate_district","state_house_district")
df <- df[,(names(df) %in% features)]

df$date <- ymd(df$date)
df$year <- year(df$date)
df$month <- month(df$date)
df$victims <- df$n_injured+df$n_killed

df <- df %>% filter(date >= as.Date("2014-01-01") & date <= as.Date("2017-12-31"))
df$weekday <- wday(df$date, label = TRUE)
df$is_weekend <- ifelse(wday(df$date) %in% c(1, 7), 1, 0)

df$incident_characteristics <- tolower(df$incident_characteristics)

# Create officer_shot and officer_killed columns
df$officer_shot <- ifelse(grepl("officer shot", df$incident_characteristics), 1, 0)
df$officer_killed <- ifelse(grepl("officer killed", df$incident_characteristics), 1, 0)

# Create officer_involved column
df$officer_involved <- ifelse(df$officer_shot == 1 | df$officer_killed == 1, 1, 0)
df$on_jan1 <- as.integer(format(df$date, "%m-%d") == "01-01")
df$on_jul4 <- as.integer(format(df$date, "%m-%d") == "07-04"|format(df$date, "%m-%d") == "07-05")
df$suspect_exact_age <- as.numeric(df$suspect_exact_age)

df_cleaned <- df %>%
  filter(suspect_age_group != "unknown", 
         suspect_exact_age != "unknown", 
         suspect_gender != "unknown")
starting_date <- min(df_cleaned$date, na.rm = TRUE)

df_cleaned$days_since_start = as.numeric(df_cleaned$date - starting_date)
df_cleaned$suspect_age_group <- ifelse(df_cleaned$suspect_age_group == "Adult 18+", 1, 0)
df_cleaned$date_num = as.numeric(df_cleaned$date)
df_cleaned$state <- as.factor(df_cleaned$state)
df_cleaned$city_or_county <- as.factor(df_cleaned$city_or_county)
df_cleaned$suspect_age_group <- as.factor(df_cleaned$suspect_age_group)
df_cleaned$suspect_gender <- ifelse(df_cleaned$suspect_gender == "Male", 1, 0)

df_cleaned=na.omit(df_cleaned)
#feature importance for predicting victims
myforest=randomForest(victims~date+state+suspect_age_group+suspect_exact_age+suspect_gender+year+month
                      +officer_shot+officer_killed+officer_involved+congressional_district+state_senate_district+
                        state_house_district+weekday+is_weekend+on_jul4+on_jan1+days_since_start, ntree=500, data=df_cleaned, importance=TRUE,  na.action = na.omit)
importance(myforest)
varImpPlot(myforest)

#model 1: predicting for future events
set.seed(661) 
# Splitting data into training and testing sets
split <- createDataPartition(df_cleaned$victims, p = 0.75, list = FALSE)
training_set <- df_cleaned[split, ]
testing_set <- df_cleaned[-split, ]

# Fit the gradient boosting model
boosted <- gbm(victims ~ date_num + days_since_start + state + suspect_age_group + suspect_gender + year + month
               + congressional_district + state_senate_district + state_house_district + weekday +  is_weekend + on_jan1,
               data = training_set,
               distribution = "gaussian",
               n.trees = 1000,
               interaction.depth = 4)

# Predict on the testing set
predictions <- predict(boosted, newdata = testing_set, n.trees = 1000)

# Calculate the accuracy or other performance metrics
# For example, using Mean Squared Error (MSE)
mse <- mean((predictions - testing_set$victims)^2)
print(mse)

plot_data <- data.frame(Index = 1:nrow(testing_set), 
                        Actual = testing_set$victims, 
                        Predicted = predictions)

# Melt the data for plotting
plot_data_long <- melt(plot_data, id.vars = "Index", variable.name = "Type", value.name = "Victims")

# Create the plot with two lines
gb_plot = ggplot(plot_data_long, aes(x = Index, y = Victims, color = Type)) +
  geom_line() +
  labs(title = "GB before Inc: Actual vs Predicted", x = "Index", y = "Number of Victims") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))

# Fit the Random Forest model
rf_model <- randomForest(victims ~ date_num + days_since_start + state + suspect_age_group + suspect_gender + year + month
                         + congressional_district + state_senate_district + state_house_district + weekday +  is_weekend + on_jan1,
                         data = training_set,
                         na.action = na.omit,
                         ntree = 1000)

# Predict on the testing set
rf_predictions <- predict(rf_model, newdata = testing_set)

# Calculate the Mean Squared Error (MSE)
rf_mse <- mean((rf_predictions - testing_set$victims)^2)
print(rf_mse)

plot_data1 <- data.frame(Index = 1:nrow(testing_set), 
                         Actual = testing_set$victims, 
                         Predicted = rf_predictions)

# Melt the data for plotting
plot_data_long1 <- melt(plot_data1, id.vars = "Index", variable.name = "Type", value.name = "Victims")

# Create the plot with two lines
rf1_plot = ggplot(plot_data_long1, aes(x = Index, y = Victims, color = Type)) +
  geom_line() +
  labs(title = "RF before Inc: Actual vs Predicted", x = "Index", y = "Number of Victims") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))
#use everything
myforest1=randomForest(victims~date+state+suspect_age_group+suspect_exact_age+suspect_gender+year+month
                       +officer_shot+officer_killed+officer_involved+congressional_district+state_senate_district+
                         state_house_district+weekday+is_weekend +days_since_start+on_jan1, ntree=500, data=training_set, importance=TRUE,  na.action = na.omit)
# Predict on the testing set
rf_predictions1 <- predict(myforest1, newdata = testing_set)
# Calculate the Mean Squared Error (MSE)
rf_mse1 <- mean((rf_predictions1 - testing_set$victims)^2)
print(rf_mse1)

plot_data2 <- data.frame(Index = 1:nrow(testing_set), 
                         Actual = testing_set$victims, 
                         Predicted = rf_predictions1)

# Melt the data for plotting
plot_data_long2 <- melt(plot_data2, id.vars = "Index", variable.name = "Type", value.name = "Victims")

# Create the plot with two lines
rf2_plot = ggplot(plot_data_long2, aes(x = Index, y = Victims, color = Type)) +
  geom_line() +
  labs(title = "RF after Inc: Actual vs Predicted", x = "Index", y = "Number of Victims") +
  theme_minimal() +
  scale_color_manual(values = c("Actual" = "blue", "Predicted" = "red"))

ggarrange(gb_plot,rf1_plot,rf2_plot,labels = c("A","B","C"),font.label = list(color="grey"))

#model2: probability of getting shot or killed
# Convert the date to a numeric format, such as the number of days since the earliest date
set.seed(661)
split_logit <- createDataPartition(df_cleaned$officer_involved, p = 0.75, list = FALSE)
training_set_logit <- df_cleaned[split_logit, ]
testing_set_logit <- df_cleaned[-split_logit, ]


# Fit logistic regression model
model <- glm(officer_involved ~  days_since_start +suspect_age_group + suspect_gender + year + month + congressional_district + state_senate_district + state_house_district, data = training_set_logit, family = binomial())


# Predict on the testing set
predictions_prob <- predict(model, newdata = testing_set_logit, type = "response")

# Binarize predictions (as logistic regression outputs probabilities)
predicted_classes <- ifelse(predictions_prob > 0.5, 1, 0)

# Calculate accuracy
accuracy <- mean(predicted_classes == testing_set_logit$officer_involved)
print(accuracy)

#model2: probability of getting killed
set.seed(661)
split_logit2 <- createDataPartition(df_cleaned$officer_killed, p = 0.75, list = FALSE)
training_set_logit2 <- df_cleaned[split_logit2, ]
testing_set_logit2 <- df_cleaned[-split_logit2, ]

# Fit logistic regression model
model2 <- glm(officer_killed ~ days_since_start + suspect_age_group + suspect_gender + year + month + congressional_district + state_senate_district + state_house_district, data = training_set_logit2, family = binomial())


# Predict on the testing set
predictions_prob2 <- predict(model2, newdata = testing_set_logit2, type = "response")

# Binarize predictions (as logistic regression outputs probabilities)
predicted_classes2 <- ifelse(predictions_prob2 > 0.5, 1, 0)

# Calculate accuracy
accuracy2 <- mean(predicted_classes2 == testing_set_logit2$officer_killed)
print(accuracy2)
