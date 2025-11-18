library(ggplot2)
library(bigrquery)
library(data.table)
library(magrittr)

# Observed data
n <- 1000
a_success_rate <- 0.05
b_success_rate <- 0.06
remaining_users <- 6000   # remaining users

# Posterior parameters (Beta(1,1) prior)
a_alpha <- 1 + a_success_rate * n
a_beta  <- 1 + (1 - a_success_rate) * n
b_alpha <- 1 + b_success_rate * n
b_beta  <- 1 + (1 - b_success_rate) * n

# Monte Carlo simulation
simulations <- 100000
pA <- rbeta(simulations, a_alpha, a_beta)
pB <- rbeta(simulations, b_alpha, b_beta)
diff <- pB - pA
credA <- quantile(pA, c(0.025, 0.975))
credB <- quantile(pB, c(0.025, 0.975))

# plot the density distributions of A and B
ggplot(data.frame(pA = pA, pB = pB)) +
  geom_density(aes(x = pA), fill = "red", alpha = 0.5) +
  geom_density(aes(x = pB), fill = "blue", alpha = 0.5) +
  geom_vline(xintercept = credA[1], color = "red", linetype = "dashed") +
  geom_vline(xintercept = credA[2], color = "red", linetype = "dashed") +
  geom_vline(xintercept = credB[1], color = "blue", linetype = "dashed") +
  geom_vline(xintercept = credB[2], color = "blue", linetype = "dashed") +
  labs(x = "Distribution of success rates", y = "Density", title = "AB test without uncertainty")

# Key metrics
prob_B_better <- mean(diff > 0)
expected_regret_per_user <- mean(pmin(0, diff))
expected_gain_per_user <- mean(pmax(0, diff))
expected_regret_total <- expected_regret_per_user * remaining_users
expected_gain_total <- expected_gain_per_user * remaining_users

cat("Pr(B > A):", prob_B_better, "\n")
# cat("A 95% CI:", credA, "\n")
# cat("B 95% CI:", credB, "\n")
cat("Expected regret (clicks):", round(expected_regret_total, 1), "\n")
cat("Expected gain (clicks):", round(expected_gain_total, 1), "\n")



# ---- TEST CAMPAIGN SIZES

query <- "WITH

campaigns_with_test_and_final_launches AS (
  SELECT customer_id, campaign_id, COUNT(DISTINCT id) AS cnt, STRING_AGG(DISTINCT type ORDER BY type) AS types
  FROM `ems-data-platform.email_launches_raw.launches`
  JOIN (SELECT account_id AS customer_id FROM `ems-account-data.unified_account_data.unified_account_data_v2` WHERE production_usage)
  USING (customer_id)
  WHERE started >= '2025-01-01'
  GROUP BY ALL
  HAVING types like '%test%' AND types like '%final%'

)

, filtered_launches AS (
  SELECT customer_id, campaign_id, id AS launch_id, type
  FROM `ems-data-platform.email_launches_raw.launches`
  JOIN campaigns_with_test_and_final_launches
  USING (customer_id, campaign_id)
)

SELECT customer_id, campaign_id, launch_id, COUNT(*) AS num_send
FROM `ems-data-platform.email_sends.sends` AS sends
JOIN filtered_launches
USING (customer_id, campaign_id, launch_id)
WHERE _PARTITIONTIME >= '2025-10-01'
GROUP BY ALL"

simplistic_query <- "
SELECT customer_id, campaign_id, launch_id, COUNT(*) AS num_send
FROM `ems-data-platform.email_sends.sends` AS sends
JOIN (SELECT customer_id, campaign_id, id AS launch_id FROM `ems-data-platform.email_launches_raw.launches` WHERE type = 'test')
USING (customer_id, campaign_id, launch_id)
WHERE _PARTITIONTIME >= '2025-10-29'
GROUP BY ALL"

bq_query_result <- bq_project_query("ems-data-playground", query)
result <- as.data.table(bq_table_download(bq_query_result))

result[, .N, by = .(group = round(num_send, -4))][order(group)][group <= 100000]

ggplot(result, aes(x = num_send)) +
  geom_histogram() +
  labs(
    x = "Number of sends",
    y = "Count",
    title = "Distribution of the number of sends"
  )

ggplot(result[num_send < 150000], aes(x = num_send)) +
  geom_histogram() +
  labs(
    x = "Number of sends",
    y = "Count",
    title = "Distribution of the number of sends"
  )