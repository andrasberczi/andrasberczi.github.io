---
layout: post
title: "Fun facts about rounding"
date: 2025-10-22
readtime: true
---

I had a discussion with a colleague about how rounding functions are different in different programming languages. I remembered, that this was quite a suprise to me a couple of years ago, when I first found out about it. But it turns out, that there is a simple reason behind it!

Everyone learns in school, that if a number 'ends' with the number 5 or higher, you round up, otherwise you round down. The same rule applies for decimal numbers as well.

This rule can be called 'rounding to the nearest even number, away from zero'. If you think about it, this is exactly what happens: you choose the number that is farther from zero.

# Rounding in R and Python

Now that we know the rule, let's try it out. Let's round 1.5 in `R` or `Python` (you can use the `round` function: `round(1.5)`). The result will be 2.

Great! Now let's calculate `round(2.5)`. The result will be 2 again. So what is happening here?

There are lot's of rules that can be applied for rounding. When we have a number exactly in the middle, we can round away from zero, but there are other options as well. One is to round to the **nearest even number**. This is the rule that is used in `R` and `Python`'s `round` function.

# But why is this rule used?

The reasoning behind this rule is to make sure there is no bias, when we are rounding a lot of numbers.

Imagine you have to calculate the average of these numbers: 1.5, 2.5, 3.5, 4.5. If you round each number to the nearest even number, you get 2, 2, 4, 4. If you round each number to the nearest odd number, you get 2, 3, 4, 5.

The average of the original numbers are 3. If you round each number to the nearest even number, the average is 3 again.
But, if you round away from zero, the average is 3.5!

Even in so simple example you can see, how bias can be introduced!

# Closing thoughts

Just to be clear, I think of this blogpost as a fun brain teaser, nothing more. I like such simple examples, which makes you think and question your intuition.

I am **not** suggesting, that you should not use the 'traditional' rounding rule anymore! Unless there are lot of numbers on the 'edge', the rounding method does not matter that much, at least in the field I am working in. But it might come handy at some point, that you know that there is a difference!
