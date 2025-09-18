---
layout: post
title: "Test your intuition on p-values - brain teaser by Ron Kohavi"
date: XXXXX - TO BE UPDATED
cover-img: ./MDE_2.png
share-img: ./MDE_2.png
thumbnail-img: ./MDE_2.png
readtime: true
---

Not long ago I saw this [A/B testing brain teaser by Ron Kohavi](
https://www.linkedin.com/posts/ronnyk_test-your-intuition-on-p-values-you-design-activity-7136272639823663104-H7u8/):

"Test your intuition on p-values?
You design an A/B test to detect a relative effect of at least 5% (MDE) to your conversion rate using industry recommended parameters: alpha=0.05 (5% type-I error) and 80% power (20% type-II error).

When the experiment finishes and reaches the planned number of users per above, you see that the treatment effect is exactly 5%.

What will be the p-value?"

I think a lot of people would say 0.05, because that's how the experiment is designed, but let's dig into it and check if it is really the case.

# Quick recap on the Minimum Detectable Effect

Obviously there is a catch and it is not too complicated either, you just need to understand what the Minimum Detectable Effect (MDE) is and how it relates to the setup of the experiment.

So what is the Minimum Detectable Effect? I actually wrote a blogpost about the MDE a while ago ([check it out!! :)](https://blog.craftlab.hu/checking-both-sides-the-minimum-detectable-effect-f34a6c0db4fb)), but in short: it is the Minimum Effect we can expect to measure (as a statistically significant effect) with a given setup for an experiment. I did write a 'footnote' however, which is key to answer the question:
"Actually this is not totally true since the MDE shows the mean difference between two samples. It is possible that the Measured Effect is slightly smaller than the MDE but our results are still statistically significant."

During hypothesis testing we are actually comparing the **distribution** of potential effects we could measure if the null or the alternative hypothesis is true. Since the MDE actually is a point value for the mean difference between two samples, it is possible that a somewhat smaller effect is still statistically significant. Turning it around: to make sure our experiment has enough power, we need to make sure the distribution for the alternative hypothesis is 'far enough' from the null distribution, which makes the MDE (the mean of the alternative distribution) even a bit further.

![MDE](../MDE_2.png)

# What is the p-value for our brain teaser?





