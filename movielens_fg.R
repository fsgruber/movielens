#######################################
# Gather data; Generate train/test sets
#######################################


#load required libraries; install if not available
if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(data.table)) install.packages("data.table")
if(!require(recosystem)) install.packages("recosystem")
if(!require(kableExtra)) install.packages("kableExtra")

#plotting theme
theme_set(theme_bw())

theme_update(
  panel.grid.major = element_blank(),
  panel.grid.minor = element_blank()
)


#assign a temp file for data download
dl <- tempfile()
download.file('http://files.grouplens.org/datasets/movielens/ml-10m.zip', dl)

#load ratings table using fred; substitute "::" with tab
ratings <- data.table::fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))


#load movies tables
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)

colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))


#combine ratings and movie table
movielens <- left_join(ratings, movies, by = "movieId")


set.seed(1, sample.kind="Rounding")

# Validation set will be 10% of MovieLens data
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


#remove users and movies from test set (validation), which are not part in the train set (edx set)
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")


# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)




#remove unused objects
rm(dl, ratings, movies, test_index, temp, movielens, removed)


#transform timestamp column into human readable format and generate week column
edx <- edx %>% 
  mutate(timestamp = lubridate::as_datetime(timestamp),
         week = lubridate::round_date(timestamp, unit = 'week'))

validation <- validation %>% 
  mutate(timestamp = lubridate::as_datetime(timestamp),
         week = lubridate::round_date(timestamp, unit = 'week'))


#dimensions of edx set
dim(edx)

#check for missing values
is.na(edx) %>% any()

#summarize number of movies and users
edx %>%
summarize(movies = n_distinct(movieId),
            users = n_distinct(userId),
          movie_times_users = movies*users) %>% 
  knitr::kable()


set.seed(1984, sample.kind = "Rounding")
userid <- sample(edx$userId, 50)

edx %>% 
  filter(userId %in% userid) %>%
  filter(movieId <=100) %>% 
  select(userId, movieId, rating) %>%
  #recode userId + movieId for plotting
  group_nest(userId) %>%
  mutate(user_id = seq(1, length(userId))) %>%
  unnest() %>%
  group_nest(movieId) %>%
  mutate(movie_id = seq(1, length(movieId))) %>%
  unnest() %>%
  ggplot(aes(movie_id, user_id, fill = as.character(rating))) +
  geom_tile() +
  scale_fill_brewer(name = 'Rating', palette = "Set3") +
  labs(title = 'Movie-User-Rating matrix',
       x = 'Movie ID', y = 'User ID') +
  theme(
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    axis.text.x = element_blank(),
    axis.text.y = element_blank()
  )

#plot movie ratings
edx %>% 
  # filter(movieId %in% c(1,2,3,4,5)) %>% 
  count(movieId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = 'black') +
  scale_x_log10() +
  labs(y = 'Number of Movies', title = "Count of movie ratings", x = 'Number of ratings')

#mean and median user ratings per movie
edx %>% 
  count(movieId) %>% 
  summarize(mean = mean(n),
            median = median(n))

#data time range
edx %>% 
  pull(timestamp) %>% 
  range()

  
#top 10 movies with most ratings
edx %>% 
  group_by(movieId, title) %>% 
  summarize(n = n()) %>% 
  ungroup() %>% 
  top_n(10, n) %>%
  arrange(desc(n)) %>% 
  knitr::kable()


  
#ratings per user
edx %>% 
    count(userId) %>% 
    summarize(mean = mean(n),
              median = median(n))

#user ratings
edx %>% 
  count(userId) %>% 
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = 'black') +
  scale_x_log10() +
  labs(title = 'User ratings',
       x = 'Number of ratings',
       y = 'Number of Users')


#weekly average
edx %>% 
  group_by(week) %>% 
  summarize(avg_rating = mean(rating)) %>%
  ggplot(aes(week, avg_rating)) +
  geom_point() +
  geom_smooth() +
  labs(title = 'Average rating per week',
       x = 'Time [weeks]', y = 'AVG rating')

#split genres; it is faster to split within a smaller table and then join by movieId
genres_split <- edx %>% 
  distinct(movieId, .keep_all = T) %>% 
  separate_rows(genres, sep = '\\|') %>% 
  select(movieId, genres)


#plot genres count
edx %>% 
  select(-genres) %>% 
  inner_join(genres_split) %>% 
  count(genres) %>% 
  # top_n(10, n) %>% 
  ggplot(aes(x = n, y = reorder(genres, n))) +
  geom_bar(stat = 'identity') +
  labs(title = 'Movie genres',
       x = 'Number of ratings',
       y = '') +
  scale_x_continuous(labels = scales::comma)



#####################
# Modelling section #
#####################


#root mean squared error loss function for model evaluation

RMSE <- function(true_ratings, predicted_ratings) {
  
  sqrt(mean( (true_ratings - predicted_ratings)^2 ))
  
}


set.seed(1985, sample.kind = 'Rounding')

#split edx into 80% train and 20% test set
test_index <- createDataPartition(edx$rating, times = 1, p = 0.2, list = FALSE)

train_set <- edx[-test_index,]
test_set <- edx[test_index,]


#remove users + movies from the test_set, which are not in the train_set
test_set <- test_set %>% 
  semi_join(train_set, by = 'movieId') %>% 
  semi_join(train_set, by = 'userId')


#first model: average only

mu <- train_set$rating %>% mean()
mu

naive_rmse <- RMSE(test_set$rating, mu)
naive_rmse


#store RMSE results in tibble
rmse_df <- tibble(
  Method = 'Average only',
  RMSE = naive_rmse
) 

rmse_df %>% knitr::kable()


#introduce bias: movie effect

movie_avg <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

# plot movie effects
qplot(b_i, data = movie_avg, bins = 10, color = I('black'))


# predict user rating using movie effects 
predictions <- test_set %>% 
  left_join(movie_avg, by = 'movieId') %>% 
  mutate(pred = mu + b_i)


rmse_df <-  rmse_df %>% 
  rbind(tibble(
    Method = 'Movie Effect Model',
    RMSE = RMSE(test_set$rating, predictions$pred)
    ))

rmse_df %>% knitr::kable()

#user effect
user_avg <- train_set %>% 
  left_join(movie_avg, by = 'movieId') %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating - mu - b_i))


predictions <- test_set %>% 
  left_join(movie_avg, by = 'movieId') %>% 
  left_join(user_avg, by = 'userId') %>% 
  mutate(pred = mu + b_i + b_u) %>% 
  pull(pred)


rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = 'Movie + User Effect Model', 
    RMSE = RMSE(test_set$rating, predictions)
    ))

rmse_df %>% knitr::kable()



#time effect

weeks_avg <- train_set %>% 
  left_join(movie_avg, by = 'movieId') %>% 
  left_join(user_avg, by = 'userId') %>% 
  group_by(week) %>% 
  summarize(b_t = mean(rating - mu - b_i - b_u))



predictions <- test_set %>% 
  left_join(movie_avg, by = 'movieId') %>% 
  left_join(user_avg, by = 'userId') %>% 
  left_join(weeks_avg, by = 'week') %>% 
  mutate(pred = mu + b_i + b_u + b_t)

rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = 'Movie + User + Week Effect Model',
    RMSE = RMSE(test_set$rating, predictions$pred)
  ))

rmse_df %>% knitr::kable()


#model regularization

lambdas <- seq(0, 10, 0.25)

regularization_wrapper <- function(trainSet, testSet, lambda = lambdas) {
  
  rmses <- sapply(lambda, function(l){
    
    mu <- trainSet$rating %>% mean()
    
    b_i <- trainSet %>% 
      group_by(movieId) %>% 
      summarize(b_i = sum(rating - mu)/(n() + l))
    
    b_u <- trainSet %>% 
      left_join(b_i, by = 'movieId') %>% 
      group_by(userId) %>% 
      summarize(b_u = sum(rating - b_i - mu)/(n() + l))
    
    b_t <- trainSet %>%
      left_join(b_i, by = 'movieId') %>%
      left_join(b_u, by = 'userId') %>%
      group_by(week) %>%
      summarize(b_t = sum(rating - mu - b_i - b_u)/ (n() + l))
    
    predictions <- testSet %>% 
      left_join(b_i, by = 'movieId') %>% 
      left_join(b_u, by = 'userId') %>% 
      left_join(b_t, by = 'week') %>%
      mutate(pred = mu + b_i + b_u + b_t) %>%
      pull(pred)
    
    return(RMSE(testSet$rating, predictions))
    
  })
  
}

#run wrapper
reg_rmses <- regularization_wrapper(train_set, test_set, lambdas)


rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = 'Regularization Movie/User/Time effect', 
    RMSE = min(reg_rmses)))

#plot lambdas 
qplot(lambdas, reg_rmses, xlab = 'Lambda', ylab = 'RMSE')

#lambda used to achieve lowest RMSE
lambdas[which.min(reg_rmses)]

rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = 'regularization movie/user/time effect', 
    RMSE = min(reg_rmses)))

rmse_df %>% knitr::kable()




#Recosystem Matrix Factorization
#https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html



recosystem_wrapper <- function(trainSet, testSet) {
set.seed(1986, sample.kind = "Rounding")
  
  train_data <- with(trainSet, data_memory(user_index = userId,
                                            item_index = movieId,
                                            rating = rating))
  
  
  test_data <- with(testSet, data_memory(user_index = userId,
                                          item_index = movieId,
                                          rating = rating))
  #initialize model
  r <- Reco()
  
  #parameter optimization
  opts <- r$tune(train_data, opts = list(dim = c(10,20,30),
                                    lrate = c(0.1, 0.2),
                                    costp_l1 = 0,
                                    costq_l1 = 0,
                                    nthread = 4,
                                    niter = 10))
  
  #train model
  r$train(train_data, opts = c(opts$min, nthread = 1, niter = 20))
  
  #predict model
  reco_pred <- r$predict(test_data, out_memory())

}

recosystem_train <- recosystem_wrapper(train_set, test_set)


rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = 'Recosystem Matrix Factorization', 
    RMSE = RMSE(test_set$rating, recosystem_train)))

rmse_df %>% knitr::kable() %>% kable_styling(latex_options = 'HOLD_position')



# Model validation: test regularization model and recosystem model

reg_rmses_val <- regularizaton_wrapper(edx, validation)

recosystem_val <- recosystem_wrapper(edx, validation)


rmse_df <- rmse_df %>% 
  rbind(tibble(
    Method = c('Regularization Model Validation', 'Recosystem Model Validation'), 
    RMSE = c(min(reg_rmses_val), RMSE(validation$rating, recosystem_val))
    ))

rmse_df %>% knitr::kable() %>% kable_styling(latex_options = 'HOLD_position')


```

