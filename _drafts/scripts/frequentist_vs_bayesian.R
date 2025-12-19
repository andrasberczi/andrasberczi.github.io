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
n <- 1000

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

# function is created below, just trying it here as well, as it seems it gives different results
paste("Ratio when B is better than A:", round(compute_posterior_prob_b_greater_than_a(
  a_successes = a_success_rate_alternative * n, a_trials = n,
  b_successes = b_success_rate_alternative * n, b_trials = n
), 4) * 100, "%")


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



###### Bayesian Decision Making: Posterior Probabilities and Bayes Factor
###### Side-by-side comparison with Frequentist approach

n <- 20

# Helper function to compute P(B > A | data) using Beta-Binomial posteriors
# With uninformative Beta(1,1) prior, posterior is Beta(1 + successes, 1 + failures)
compute_posterior_prob_b_greater_than_a <- function(a_successes, a_trials, b_successes, b_trials, num_samples = 10000) {
  # Uninformative Beta(1,1) prior
  alpha_prior <- 1
  beta_prior <- 1

  # Posterior parameters
  alpha_a <- alpha_prior + a_successes
  beta_a <- beta_prior + (a_trials - a_successes)
  alpha_b <- alpha_prior + b_successes
  beta_b <- beta_prior + (b_trials - b_successes)

  # Sample from posterior distributions
  posterior_a <- rbeta(num_samples, alpha_a, beta_a)
  posterior_b <- rbeta(num_samples, alpha_b, beta_b)

  # P(B > A | data)
  mean(posterior_b > posterior_a)
}

# Helper function to compute Bayes Factor for one-sided test: H1 (B > A) vs H0 (B = A)
# Using Beta-Binomial with uninformative priors
compute_bayes_factor <- function(a_successes, a_trials, b_successes, b_trials) {
  # Uninformative Beta(1,1) prior
  alpha_prior <- 1
  beta_prior <- 1

  # H0: B = A (pooled estimate)
  total_successes <- a_successes + b_successes
  total_trials <- a_trials + b_trials
  pooled_rate <- total_successes / total_trials

  # Marginal likelihood under H0 (pooled model)
  # Using Beta-Binomial: Beta(alpha + successes, beta + failures) / Beta(alpha, beta)
  log_likelihood_h0 <- lchoose(a_trials, a_successes) +
                       lchoose(b_trials, b_successes) +
                       lbeta(alpha_prior + total_successes, beta_prior + total_trials - total_successes) -
                       lbeta(alpha_prior, beta_prior)

  # H1: B > A (independent estimates)
  # Marginal likelihood under H1 (independent model)
  log_likelihood_h1 <- lchoose(a_trials, a_successes) +
                       lchoose(b_trials, b_successes) +
                       lbeta(alpha_prior + a_successes, beta_prior + a_trials - a_successes) -
                       lbeta(alpha_prior, beta_prior) +
                       lbeta(alpha_prior + b_successes, beta_prior + b_trials - b_successes) -
                       lbeta(alpha_prior, beta_prior)

  # For one-sided test, we need to account for the constraint B > A
  # Approximate using Savage-Dickey ratio or compute restricted marginal likelihood
  # Simplified approach: use ratio of marginal likelihoods with correction for one-sidedness
  # BF = P(data | H1) / P(data | H0), but for one-sided we want P(data | B > A) / P(data | B = A)

  # For computational simplicity, we'll use the ratio and note that for one-sided,
  # we'd typically need to integrate over the constrained space
  # This is an approximation that works well when the effect is clear
  bf_ratio <- exp(log_likelihood_h1 - log_likelihood_h0)

  # For one-sided test, we can also compute P(B > A | data) and use that
  # A more accurate BF for one-sided would require integration, but this gives intuition
  return(bf_ratio)
}

# Helper function for frequentist one-sided test
compute_frequentist_pvalue <- function(a_successes, a_trials, b_successes, b_trials) {
  # Two-proportion z-test (one-sided: H1: B > A)
  p_a <- a_successes / a_trials
  p_b <- b_successes / b_trials
  p_pooled <- (a_successes + b_successes) / (a_trials + b_trials)

  se_pooled <- sqrt(p_pooled * (1 - p_pooled) * (1/a_trials + 1/b_trials))
  z_stat <- (p_b - p_a) / se_pooled

  # One-sided p-value
  1 - pnorm(z_stat)
}

# Simulate decisions for multiple experiments
num_experiments <- 1000
decision_results_null <- data.table(
  experiment = 1:num_experiments,
  a_successes = rbinom(num_experiments, n, a_success_rate_null),
  b_successes = rbinom(num_experiments, n, b_success_rate_null),
  a_trials = n,
  b_trials = n
)

decision_results_alternative <- data.table(
  experiment = 1:num_experiments,
  a_successes = rbinom(num_experiments, n, a_success_rate_alternative),
  b_successes = rbinom(num_experiments, n, b_success_rate_alternative),
  a_trials = n,
  b_trials = n
)

# Compute metrics for each experiment
decision_results_null[, `:=`(
  p_value = mapply(compute_frequentist_pvalue, a_successes, a_trials, b_successes, b_trials),
  posterior_prob = mapply(compute_posterior_prob_b_greater_than_a, a_successes, a_trials, b_successes, b_trials),
  bayes_factor = mapply(compute_bayes_factor, a_successes, a_trials, b_successes, b_trials)
)]

decision_results_alternative[, `:=`(
  p_value = mapply(compute_frequentist_pvalue, a_successes, a_trials, b_successes, b_trials),
  posterior_prob = mapply(compute_posterior_prob_b_greater_than_a, a_successes, a_trials, b_successes, b_trials),
  bayes_factor = mapply(compute_bayes_factor, a_successes, a_trials, b_successes, b_trials)
)]

# Decision rules
# Frequentist: reject H0 (choose B) if p-value < 0.05
# Bayesian (Posterior Prob): choose B if P(B > A | data) > 0.95
# Bayesian (Bayes Factor): choose B if BF > 3 (moderate evidence) or > 10 (strong)

decision_results_null[, `:=`(
  frequentist_decision = p_value < 0.05,
  bayesian_postprob_decision = posterior_prob > 0.95,
  bayesian_bf_decision_moderate = bayes_factor > 3,
  bayesian_bf_decision_strong = bayes_factor > 10
)]

decision_results_alternative[, `:=`(
  frequentist_decision = p_value < 0.05,
  bayesian_postprob_decision = posterior_prob > 0.95,
  bayesian_bf_decision_moderate = bayes_factor > 3,
  bayesian_bf_decision_strong = bayes_factor > 10
)]

# Summary statistics: Decision rates
cat("\n=== NULL HYPOTHESIS TRUE (B = A) ===\n")
cat("Frequentist (p < 0.05):", round(mean(decision_results_null$frequentist_decision) * 100, 2), "%\n")
cat("Bayesian Posterior Prob (P(B>A) > 0.95):", round(mean(decision_results_null$bayesian_postprob_decision) * 100, 2), "%\n")
cat("Bayesian BF (BF > 3):", round(mean(decision_results_null$bayesian_bf_decision_moderate) * 100, 2), "%\n")
cat("Bayesian BF (BF > 10):", round(mean(decision_results_null$bayesian_bf_decision_strong) * 100, 2), "%\n")

cat("\n=== ALTERNATIVE HYPOTHESIS TRUE (B > A) ===\n")
cat("Frequentist (p < 0.05):", round(mean(decision_results_alternative$frequentist_decision) * 100, 2), "%\n")
cat("Bayesian Posterior Prob (P(B>A) > 0.95):", round(mean(decision_results_alternative$bayesian_postprob_decision) * 100, 2), "%\n")
cat("Bayesian BF (BF > 3):", round(mean(decision_results_alternative$bayesian_bf_decision_moderate) * 100, 2), "%\n")
cat("Bayesian BF (BF > 10):", round(mean(decision_results_alternative$bayesian_bf_decision_strong) * 100, 2), "%\n")

# Visualization: Distribution of metrics
# Null hypothesis
ggplot(decision_results_null) +
  geom_histogram(aes(x = p_value), bins = 50, fill = "red", alpha = 0.5, position = "identity") +
  geom_histogram(aes(x = posterior_prob), bins = 50, fill = "blue", alpha = 0.5, position = "identity") +
  geom_vline(xintercept = 0.05, color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 0.95, color = "blue", linetype = "dashed", linewidth = 1) +
  labs(
    x = "Value",
    y = "Frequency",
    title = "Null Hypothesis: Distribution of p-values (red) and P(B>A|data) (blue)"
  ) +
  scale_x_continuous(limits = c(0, 1))

# Alternative hypothesis
ggplot(decision_results_alternative) +
  geom_histogram(aes(x = p_value), bins = 50, fill = "red", alpha = 0.5, position = "identity") +
  geom_histogram(aes(x = posterior_prob), bins = 50, fill = "blue", alpha = 0.5, position = "identity") +
  geom_vline(xintercept = 0.05, color = "red", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 0.95, color = "blue", linetype = "dashed", linewidth = 1) +
  labs(
    x = "Value",
    y = "Frequency",
    title = "Alternative Hypothesis: Distribution of p-values (red) and P(B>A|data) (blue)"
  ) +
  scale_x_continuous(limits = c(0, 1))

# Bayes Factor distribution (log scale for better visualization)
ggplot(decision_results_null) +
  geom_histogram(aes(x = log10(bayes_factor)), bins = 50, fill = "green", alpha = 0.5) +
  geom_vline(xintercept = log10(3), color = "green", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = log10(10), color = "green", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  labs(
    x = "log10(Bayes Factor)",
    y = "Frequency",
    title = "Null Hypothesis: Distribution of Bayes Factors"
  )

ggplot(decision_results_alternative) +
  geom_histogram(aes(x = log10(bayes_factor)), bins = 50, fill = "green", alpha = 0.5) +
  geom_vline(xintercept = log10(3), color = "green", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = log10(10), color = "green", linetype = "dashed", linewidth = 1) +
  geom_vline(xintercept = 0, color = "black", linetype = "dashed") +
  labs(
    x = "log10(Bayes Factor)",
    y = "Frequency",
    title = "Alternative Hypothesis: Distribution of Bayes Factors"
  )

# Side-by-side comparison: Agreement between methods
cat("\n=== AGREEMENT BETWEEN METHODS (NULL HYPOTHESIS) ===\n")
cat("Frequentist & Posterior Prob agree:",
    round(mean(decision_results_null$frequentist_decision == decision_results_null$bayesian_postprob_decision) * 100, 2), "%\n")
cat("Frequentist & BF (moderate) agree:",
    round(mean(decision_results_null$frequentist_decision == decision_results_null$bayesian_bf_decision_moderate) * 100, 2), "%\n")
cat("Posterior Prob & BF (moderate) agree:",
    round(mean(decision_results_null$bayesian_postprob_decision == decision_results_null$bayesian_bf_decision_moderate) * 100, 2), "%\n")

cat("\n=== AGREEMENT BETWEEN METHODS (ALTERNATIVE HYPOTHESIS) ===\n")
cat("Frequentist & Posterior Prob agree:",
    round(mean(decision_results_alternative$frequentist_decision == decision_results_alternative$bayesian_postprob_decision) * 100, 2), "%\n")
cat("Frequentist & BF (moderate) agree:",
    round(mean(decision_results_alternative$frequentist_decision == decision_results_alternative$bayesian_bf_decision_moderate) * 100, 2), "%\n")
cat("Posterior Prob & BF (moderate) agree:",
    round(mean(decision_results_alternative$bayesian_postprob_decision == decision_results_alternative$bayesian_bf_decision_moderate) * 100, 2), "%\n")

# Example: Single experiment analysis
example_experiment_null <- decision_results_null[1]
example_experiment_alt <- decision_results_alternative[which.max(decision_results_alternative$posterior_prob)[1]]

cat("\n=== EXAMPLE EXPERIMENT (NULL HYPOTHESIS) ===\n")
cat("A successes:", example_experiment_null$a_successes, "out of", example_experiment_null$a_trials, "\n")
cat("B successes:", example_experiment_null$b_successes, "out of", example_experiment_null$b_trials, "\n")
cat("Frequentist p-value:", round(example_experiment_null$p_value, 4), "-> Decision:", ifelse(example_experiment_null$frequentist_decision, "Choose B", "Choose A"), "\n")
cat("Posterior P(B > A | data):", round(example_experiment_null$posterior_prob, 4), "-> Decision:", ifelse(example_experiment_null$bayesian_postprob_decision, "Choose B", "Choose A"), "\n")
cat("Bayes Factor:", round(example_experiment_null$bayes_factor, 2), "-> Decision (BF>3):", ifelse(example_experiment_null$bayesian_bf_decision_moderate, "Choose B", "Choose A"), "\n")

cat("\n=== EXAMPLE EXPERIMENT (ALTERNATIVE HYPOTHESIS) ===\n")
cat("A successes:", example_experiment_alt$a_successes, "out of", example_experiment_alt$a_trials, "\n")
cat("B successes:", example_experiment_alt$b_successes, "out of", example_experiment_alt$b_trials, "\n")
cat("Frequentist p-value:", round(example_experiment_alt$p_value, 4), "-> Decision:", ifelse(example_experiment_alt$frequentist_decision, "Choose B", "Choose A"), "\n")
cat("Posterior P(B > A | data):", round(example_experiment_alt$posterior_prob, 4), "-> Decision:", ifelse(example_experiment_alt$bayesian_postprob_decision, "Choose B", "Choose A"), "\n")
cat("Bayes Factor:", round(example_experiment_alt$bayes_factor, 2), "-> Decision (BF>3):", ifelse(example_experiment_alt$bayesian_bf_decision_moderate, "Choose B", "Choose A"), "\n")




