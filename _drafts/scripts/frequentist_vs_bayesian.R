library(ggplot2)
library(data.table)
library(magrittr)
library(pwr)

num_simulations <- 100000
n <- 1000

###### Null hypothesis is true
a_success_rate_null <- 0.3
b_success_rate_null <- 0.3

a_standard_error_null <- sqrt(a_success_rate_null * (1 - a_success_rate_null) / n)

a_simulations_null <- rbinom(num_simulations, n, a_success_rate_null) / n
b_simulations_null <- rbinom(num_simulations, n, b_success_rate_null) / n

ggplot(data.table(a_simulations = a_simulations_null, b_simulations = b_simulations_null)) +
  geom_function(fun = function(x) dnorm(x, mean = a_success_rate_null, sd = a_standard_error_null), color = "red", size = 1.5) +
  geom_density(aes(x = a_simulations), fill = "red", alpha = 0.5) +
  geom_vline(xintercept = a_success_rate_null, color = "red", linetype = "dashed") +
  geom_vline(xintercept = a_success_rate_null + 1.96 * a_standard_error_null, color = "red") +
  geom_vline(xintercept = a_success_rate_null - 1.96 * a_standard_error_null, color = "red") +
  geom_density(aes(x = b_simulations), fill = "blue", alpha = 0.5) +
  geom_vline(xintercept = b_success_rate_null, color = "blue", linetype = "dashed") +
  labs(
    x = "Number of successes",
    y = "Density",
    title = "Distribution of success rates"
  )

# Bayesian approach: choose B if B's posterior is higher than A's
paste("Ratio when B is better than A:", round(mean(b_simulations_null > a_simulations_null), 4) * 100, "%")

paste(
    "Ratio when we choose B based on statistical significance:",
    round(mean(b_simulations_null > (a_success_rate_null + qnorm(0.975) * a_standard_error_null)), 4) * 100,
    "%"
)

### Same with difference in means

b_standard_error_null <- sqrt(b_success_rate_null * (1 - b_success_rate_null) / n)
pooled_standard_error_null <- sqrt(a_standard_error_null^2 + b_standard_error_null^2)
difference_in_means_null <- b_simulations_null - a_simulations_null

ggplot(data.table(difference_in_means = difference_in_means_null)) +
  geom_function(fun = function(x) dnorm(x, mean = 0, sd = pooled_standard_error_null), color = "blue") +
  geom_density(aes(x = difference_in_means), fill = "blue", alpha = 0.5) +
  geom_vline(xintercept = 0, color = "blue", linetype = "dashed") +
  geom_vline(xintercept = qnorm(0.025) * pooled_standard_error_null, color = "blue") +
  geom_vline(xintercept = qnorm(0.975) * pooled_standard_error_null, color = "blue") +
  labs(
    x = "Difference in means",
    y = "Density",
    title = "Distribution of difference in means"
  )


paste("Ratio when B is better than A:", round(mean(difference_in_means_null > 0), 4) * 100, "%")

paste(
    "Ratio when we choose B based on statistical significance:",
    round(mean(difference_in_means_null > qnorm(0.975) * pooled_standard_error_null), 4) * 100,
    "%"
)


###### Alternative hypothesis is true

###### Calculate sample size for a given power and effect size
a_success_rate_alternative <- 0.30
b_success_rate_alternative <- 0.31

effect_size <- b_success_rate_alternative - a_success_rate_alternative
result <- power.prop.test(p1 = a_success_rate_alternative, p2 = b_success_rate_alternative, sig.level = 0.05, power = 0.8)
n <- round(result$n)

# OVERRIDE N to see what decisions would we make with smaller samples?
n <- 100

a_standard_error_alternative <- sqrt(a_success_rate_alternative * (1 - a_success_rate_alternative) / n)
b_standard_error_alternative <- sqrt(b_success_rate_alternative * (1 - b_success_rate_alternative) / n)
pooled_standard_error_alternative <- sqrt(a_standard_error_alternative^2 + b_standard_error_alternative^2)

a_simulations_alternative <- rbinom(num_simulations, n, a_success_rate_alternative) / n
b_simulations_alternative <- rbinom(num_simulations, n, b_success_rate_alternative) / n
difference_in_means_alternative <- b_simulations_alternative - a_simulations_alternative

ggplot(data.table(a_simulations = a_simulations_alternative, b_simulations = b_simulations_alternative)) +
  # A Success Rate
  geom_function(fun = function(x) dnorm(x, mean = a_success_rate_alternative, sd = a_standard_error_alternative), color = "red", size = 1.5) +
  geom_density(aes(x = a_simulations), fill = "red", alpha = 0.5) +
  geom_vline(xintercept = a_success_rate_alternative, color = "red", linetype = "dashed") +
  geom_vline(xintercept = a_success_rate_alternative + qnorm(0.025) * a_standard_error_alternative, color = "red") +
  geom_vline(xintercept = a_success_rate_alternative + qnorm(0.975) * a_standard_error_alternative, color = "red") +
  # B Success Rate
  geom_density(aes(x = b_simulations), fill = "blue", alpha = 0.5) +
  geom_function(fun = function(x) dnorm(x, mean = b_success_rate_alternative, sd = b_standard_error_alternative), color = "blue", size = 1.5) +
  geom_vline(xintercept = b_success_rate_alternative, color = "blue", linetype = "dashed") +
  labs(
    x = "Number of successes",
    y = "Density",
    title = "Distribution of success rates"
  )

paste("Ratio when B is better than A:", round(mean(b_simulations_alternative > a_simulations_alternative), 4) * 100, "%")
paste("Ratio when B is better than A:", round(mean(difference_in_means_alternative > 0), 4) * 100, "%")

paste(
    "Ratio when we choose B based on statistical significance:",
    round(mean(difference_in_means_alternative > qnorm(0.975) * pooled_standard_error_alternative), 4) * 100,
    "%"
)

ggplot(data.table(difference_in_means = difference_in_means_alternative)) +
  geom_function(fun = function(x) dnorm(x, mean = 0, sd = pooled_standard_error_alternative),
                color = "red", linetype = "dashed", alpha = 0.7) +
  geom_function(fun = function(x) dnorm(x, mean = effect_size, sd = pooled_standard_error_alternative),
                color = "blue", linetype = "dashed", alpha = 0.7) +
  geom_density(aes(x = difference_in_means), fill = "blue", alpha = 0.5) +
  geom_vline(xintercept = 0, color = "red", linetype = "dashed") +
  geom_vline(xintercept = effect_size, color = "blue", linetype = "dashed") +
  geom_vline(xintercept = qnorm(0.025) * pooled_standard_error_alternative, color = "red", linewidth = 1) +
  geom_vline(xintercept = qnorm(0.975) * pooled_standard_error_alternative, color = "red", linewidth = 1) +
  geom_vline(xintercept = effect_size + qnorm(0.2) * pooled_standard_error_alternative, color = "green", linewidth = 1) +
  labs(
    x = "Difference in means",
    y = "Density",
    title = "Distribution of difference in means (Null vs Alternative)"
  )

## Cost of choice in Bayesian approach

# NULL

# Expected regret per user
mean(pmin(0, difference_in_means_null))
# Expected gain per user
mean(pmax(0, difference_in_means_null))

# ALTERNATIVE

# Expected regret per user
mean(pmin(0, difference_in_means_alternative))
# Expected gain per user
mean(pmax(0, difference_in_means_alternative))

# Expected regret per user as a percentage of the effect size
mean(pmin(0, difference_in_means_alternative)) / effect_size
# Expected gain per user as a percentage of the effect size
mean(pmax(0, difference_in_means_alternative)) / effect_size

# expected_regret_total <- expected_regret_per_user * remaining_users
# expected_gain_total <- expected_gain_per_user * remaining_users