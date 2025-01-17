# How Transformer Models Work: Breaking Down the Magic of GPT

## Introduction
I wanted to get a deeper understanding on how GPT models work. I have watched Andrej Karpaty's [amazing Youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) on this subject.
In this blogpost I will write down what I have learned. I do this for 2 reasons:
1. it is a good way to check if I have really understood the video,
2. I want to share this knowledge with the world.

So how do transformer models, like those powering GPT, work? While the final models may seem like black boxes, their underlying mechanisms are surprisingly structured and elegant.

Of course I don't want to go as deep as Andrej does in his video, so if you want to get a real deep understanding, I really suggest to watch his video! I will just highlight the main steps and ellaborate on some parts, which I think are interesting.

## Understanding the Problem: Context and Sequence in Language Modeling

Language modeling involves predicting the next word (or character) in a sequence based on prior context. For instance, given the phrase “To be or not to”, a good model should predict “be” as the next word. Traditional models, like n-grams, struggle to capture long-range dependencies effectively. These models usually take the last n words/characters/tokens as an input and predict the next one based on all these inputs.

Transformers address this limitation by focusing on the relationship between all elements in a sequence through their self-attention mechanism.

In this blogpost I am going to use the same example Andrej uses in his video. Our task is to implement a model, which can generate text, based on some input (text). We are going to  train the model based on a txt file, which [contains text from Shakespeare's work](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).

I will also provide some code in the hope, that it helps understanding the concepts better. However, it is not a full implementation, if you want a full code, with all the dependencies to run it, I suggest to check out Andrej's github repository.


## Building on the foundation of Bigram Model.

Although I have just mentioned it's shortcomings, to keep thins simple, we will start with a traditional **bigram model**, where we will predict the next character based only on the current one. You saw right: we will predict the next character. This is just for simplicity. In a real world scenario, we would predict the next token or next word, but for now, we will stick with characters.

As a first step, we need to encode the characters into numerical values. We will use a simple dictionary for this, so basically assign a number to each letter in the abc. Eg.: {'a': 0, 'b': 1, 'c': 2, ...}. These number representations will be used as input for our model. (See example code in collapsabel details below.)

<details>

```python
# input_file_path is the path to the txt file, which contains the text we want to train our model on.
# It contains text from Shakespeare's work, as mentioned in the block above.
with open(input_file_path, "r", encoding="utf-8") as f:
  text = f.read()

# Character-level encoding
# The numbers are the indices of the characters in the `chars` list
# So basically we are converting the characters to numbers by alphabetical order
chars = sorted(list(set(text)))
stoi = {char: i for i, char in enumerate(chars)}
itos = {i: char for i, char in enumerate(chars)}

encode = lambda s: [stoi[char] for char in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
```
</details>

Then we start to build our model. The first model will be very simple, it will just consist of an embedding layer.
The embedding layer will map the input characters to vectors. This vector will be the representation of a sequence of characters. It will be used during the training process: the model will learn to predict the next character based on this vector representation.

```python
class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      logits = logits.view(B * T, C)
      targets = targets.view(B * T)
      loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, loss = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1)
      idx = torch.cat([idx, idx_next], dim=1)
    return idx
```

In the code above we have the `forward` and the `generate` methods. The `forward` method is used during the training process. It takes the input characters and the target characters as input and returns the logits and the loss. We can use this information to update the weights of the model during the training process.

<details>

```python
for steps in range(eval_iters):
    if steps % print_iter == 0:
        losses = estimate_loss()
        print(f"step: {steps}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # The get_batch function generates a batch of input and target sequences from either training or validation data by randomly sampling starting indices and extracting sequences of a specified length.
    # This is used to prepare data for training or validating sequence models.
    xb, yb = get_batch("train")

    # evaluate the model on the batch
    logits, loss = model(xb, yb)

    # update the model's parameters
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```
</details>

The `generate` method is used to generate text. It takes the input characters and the number of characters we want to generate as input and returns the generated characters.

## Step 2: Introducing the Self-Attention Mechanism

Now you might be able to imagine, that this process is too simple to work well.

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

