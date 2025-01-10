# How Transformer Models Work: Breaking Down the Magic of GPT

## Introduction
I wanted to get a deeper understanding on how GPT models work. I have watched Andrej Karpaty's [amazing Youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) on this subject.
In this blogpost I will write down what I have learned. I do this for 2 reasons:
1. it is a good way to check if I have really understood the video,
2. I want to share this knowledge with the world.

Of course I don't want to go as deep as Andrej does in his video, so if you want to get a real deep understanding, I really suggest to watch his video!

So how do transformer models, like those powering GPT, work? While the final models may seem like black boxes, their underlying mechanisms are surprisingly structured and elegant. In this blog post, I’ll break down the essential components of transformers and explain how they work.

In the video Andrej explains the mechanism through code, where he implements a model, which can generate text. The model is trained on a txt file, which [contains text from Shakespeare's work](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt). I will use the same example

## Understanding the Problem: Context and Sequence in Language Modeling

Language modeling involves predicting the next word (or character) in a sequence based on prior context. For instance, given the phrase “To be or not to”, a good model should predict “be” as the next word.

Traditional models, like n-grams, struggle to capture long-range dependencies effectively. These models usually take the last n words/characters/tokens as an input, but fails to grasp the importance of the words, when predicting the next one.

Transformers address this limitation by focusing on the relationship between all elements in a sequence through their self-attention mechanism.

## Building on the foundation of Bigram Model.

We began our journey by implementing a simple **bigram model**, which predicts the next character based only on the current one.

Models can work with numbers, so we need to 'convert' text to numbers.
For this, we tokenize the text at the character level, converting each character into a unique numerical ID.
2. Created an **embedding layer** to map these IDs to vectors, which are richer numerical representations.
3. Predicted the next character using a simple lookup table.

For example, if the input is “To”, the model predicts “ ” (space) as the most likely next character based on patterns in the training data.

While straightforward, the bigram model lacks the ability to incorporate context beyond the immediate previous character, leading to incoherent text generation.

## Step 2: Introducing the Self-Attention Mechanism

To overcome the limitations of the bigram model, we introduced **self-attention**, the cornerstone of transformers. Self-attention enables each token in the sequence to dynamically weigh the importance of other tokens when making predictions.

### How Self-Attention Works:
1. **Queries, Keys, and Values:** Each token emits a query (“What am I looking for?”), a key (“What do I represent?”), and a value (“What information do I carry?”).
2. **Dot Products:** The query of one token is compared to the keys of all other tokens, generating a set of scores that represent “relevance.”
3. **Softmax Normalization:** These scores are normalized to probabilities, ensuring they sum to 1.
4. **Weighted Sum:** The values are aggregated using these probabilities, producing a context-aware representation for each token.

For instance, when processing the sequence “To be or not to,” the self-attention mechanism might focus more on the second “to” when predicting the next word after “not,” capturing the repetition.

## Step 3: Multi-Head Attention for Richer Representations

**Multi-head attention** extends self-attention by running multiple attention mechanisms in parallel. Each head learns to focus on different aspects of the sequence—such as syntax, semantics, or specific recurring patterns.

For our Shakespearean text example, one head might focus on rhythmic patterns (e.g., iambic pentameter), while another emphasizes grammatical structure.

## Step 4: Transformer Blocks – Putting It All Together

A single **transformer block** integrates:

1. **Multi-head attention:** Capturing diverse relationships in the sequence.
2. **Feedforward layers:** Applying non-linear transformations to enrich representations.
3. **Residual connections:** Adding the input to the output of each layer to stabilize gradients and preserve information.
4. **Layer normalization:** Ensuring numerical stability by normalizing inputs within each layer.

By stacking multiple transformer blocks, the model iteratively refines its understanding of the input, enabling it to generate more coherent and contextually accurate text.

## Step 5: Training the Model

The model is trained by:

1. Defining a loss function—typically cross-entropy—that measures how well the predicted sequence matches the actual sequence.
2. Using backpropagation to compute gradients of the loss with respect to model parameters.
3. Optimizing parameters using algorithms like AdamW.

For our Shakespearean text, training involves exposing the model to sequences of characters and adjusting its parameters to minimize the prediction error. Over time, the model learns patterns like word structures and stylistic nuances.

## Challenges in Understanding and Implementing Transformers

Building and training a transformer model involves overcoming several challenges:

- **Numerical Stability:** Techniques like scaling dot products and masking irrelevant parts of the sequence are critical for preventing issues like exploding gradients.
- **Efficient Computation:** Matrix multiplications are key to speeding up operations, especially in attention mechanisms.
- **Hyperparameter Tuning:** Choosing the right number of attention heads, embedding dimensions, and learning rates significantly impacts performance.

## Scaling Transformers (Briefly)

While we focused on a small-scale implementation, real-world models like GPT-3 scale these principles to billions of parameters. Key differences include:

- **Datasets:** GPT-3 is trained on massive datasets, far beyond the size of Shakespeare’s works.
- **Hardware:** Training such models requires thousands of GPUs working in parallel.
- **Fine-Tuning:** These models are adapted to specific tasks using supervised and reinforcement learning techniques.

## Conclusion

Transformers are a marvel of modern machine learning, combining simple building blocks into a sophisticated system for capturing context and generating coherent text. By building a small-scale GPT-like model, we’ve explored the fundamental components—from attention to transformer blocks—and seen how they come together to solve language modeling tasks.

If you’re intrigued, I encourage you to dive into the code, experiment with different datasets, and tweak the architecture. Understanding transformers from the ground up is not only rewarding but also equips you with tools to tackle cutting-edge challenges in NLP and beyond.

