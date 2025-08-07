---
layout: post
title:  "What mean can mean?"
date:   2025-08-07
# cover-img: ./MDE.png
# share-img: ./mean.png
# thumbnail-img: ./MDE.png
readtime: true
---

If you ask a person to calculate the average of two numbers, like 10 and 20, it's not going to be a problem for them. I would guess that 99.9% of them would say it's 15.

This answer is correct of course, but few remembers their high school math, that the average or the mean can be calculated in multiple ways. In this blogpost I want to do a quick dive on why and when to use the harmonic mean, instead of the arithmetic mean. There are other kind of means as well, but I won't be covering them. If you are interested, [check them out](https://en.wikipedia.org/wiki/Mean) and think through when it's worth using them!

# Example: what your average speed?

Let's say you have to travel 200 km. The first 100 km you go on the highway and go with 130 km/h. The second 100 km you go in the city and you travel with 50 km/h. So what is your average speed?

If you use the arithmetic mean you would get:

```
(130 + 50) / 2 = 90 km/h
```

But let's think it through.
How much doest the first 100 km take? You travel 100 km with 130 km/h speed, so it takes 100 / 130 = 0.769 hours.

We can calculat the second 100 km similiarly: it takes 100 / 50 = 2 hours.

Alltogether, you travel 200 km and it takes you 2.769 hours. So your average speed for the whole trip is 200 / 2.769 = 72.23 km/h. That is not 90 km / h, as the arithmetic mean would suggest!

So what happened here? Well, here is when the harmonic mean comes into play.

Simply put: harmonic mean can be used, when we are dealing with ratios (like speed, which is distance / time). But I would add an important 'disclaimer' here: don't just use harmonic mean every time you see a ratio. It also depends on the question you want to answer and how it is 'connected' to the ratio.

A ratio of course has a numerator and denominator. Harmonic mean should be used, when you want to compare the ratio with the same numerator and the denominator is the 'variable' part. If you want to compare the ratio with the same denominator and the numerator is the 'variable' part, you should use the arithmetic mean.

This might sound a bit complicated, so let me give you an example.

In the question above we cared about the average speed over some *distance*, which is the numerator. The 'variable' part was the time spent travelling that distance, given the speed.

If I would have asked: "You travel 1 hour with 130 km/h and 1 hour with 50 km/h, what is your average speed?", then using the arithmetic mean would be correct!

(You can easily double check: You travel 130 km in the first hour and 50 km in the second hour, so you travel 180 km in 2 hours. This way you get 90 km/h.)

# What is the harmonic mean?

The formula for the harmonic mean is:

$$
\text{Harmonic Mean} = \frac{n}{\frac{1}{x_1} + \frac{1}{x_2} + \cdots + \frac{1}{x_n}}
$$

Where $n$ is the number of values and $x_1, x_2, \ldots, x_n$ are the values.

So in the case of the example above, we would calculate the harmonic mean as:

$$
\frac{2}{\frac{1}{130} + \frac{1}{50}} = 72.23 \text{ km/h}
$$

Just for fun, let's dig a bit deeper and try to understand why does it make sense to use the harmonic mean instead of the arithmetic mean? What makes this the correct way to calculate the average speed?

Going back to the example, we want to calculate how much time does it take for each part of the trip with different speed? We know the distance of each part, so that is fixed, but the speed and thus the time to take the travel changes. You can write this as:

$time_1 = \frac{distance}{speed_1}$

$time_2 = \frac{distance}{speed_2}$

($total distance = distance + distance$)

$\overline{speed} = \frac{distance + distance}{time_1 + time_2}$

$\overline{speed} = \frac{2 \cdot distance}{time_1 + time_2}$

$\overline{speed} = \frac{2 \cdot distance}{\frac{distance}{speed_1} + \frac{distance}{speed_2}}$

simplifing with distance:

$\overline{speed} = \frac{2}{\frac{1}{speed_1} + \frac{1}{speed_2}}$

## Where do we use this in data science?

Probably the most famous example is the calculation of the F1 score.

Here we want to calculate the average of precision and recall. Precision is the number of true positives divided by the number of true positives and false positives. Recall is the number of true positives divided by the number of true positives and false negatives. In both cases the numerator is the same, but we divide by a different denominator. Thus, we use the harmonic mean.

(F1 score is usually written as $2 \cdot \frac{precision \cdot recall}{precision + recall}$, but this is actually the harmonic mean, just written a bit differently. You can easily rearrange to get ${F1} = \frac{2}{\frac{1}{precision} + \frac{1}{recall}}$)

# Conclusion

I really enjoyed writing this blogpost. It made me think over how the harmonic mean works and why I shouldn't just default to the arithmetic mean without thinking.

I hope it is useful for you as well!
