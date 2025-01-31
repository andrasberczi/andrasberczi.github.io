# How Transformer Models Work: Breaking Down the Magic of GPT

## Introduction
I wanted to get a deeper understanding on how GPT models work. I have watched Andrej Karpaty's [amazing Youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) on this.
In this blogpost I will write what I have learned. I do this for 2 reasons:
1. it is a good way to check, if I have really understood the video,
2. I want to preserve this knowledge for my future self,
3. I want to share this knowledge with the world.

So how do transformer models, like those powering GPT, work? While the final models may seem like black boxes, their underlying mechanisms are surprisingly structured and elegant.

Of course I don't want to go as deep as Andrej does in his video, so if you want to get a real deep understanding, I really suggest to watch his video! I will just highlight the main steps and ellaborate on some parts, which I think are interesting.

I will also provide some code in the hope, that it helps understanding the concepts better. However, it is not a full implementation, if you want a full code, with all the dependencies to run it, I suggest to check out Andrej's github repository.

## Understanding the Problem: Context and Sequence in Language Modeling

Our goal at the end of the day is to generate text that help you achieve your task, let that be summarising, translating or having a discussion with a chatbot. There is however two categories, which are essentially different from "timing" perspective. There are the tasks, where we want to continue the text, based on the already existing one and there are the tasks where we have a "finished" text and we want to generate a new one. An example for the first one is talking to a chatbot, where you want to continue the conversation. An example for the second one is summarising a text. In this blogpost we will focus on the first one.

So Language modeling for these tasks involve predicting the next word (or character) in a sequence based on prior context. For instance, given the phrase “To be or not to”, a good model should predict “be” as the next word. There are traditional models, like n-grams (eg.: `bigram`), but they struggle to capture long-range dependencies effectively. These models usually take the last n words/characters/tokens as an input and predict the next one based on all these inputs.

Transformers address this limitation by focusing on the relationship between all elements in a sequence through their self-attention mechanism.

In this blogpost I am going to use the same example Andrej uses in his video. Our task is to implement a model, which can generate text, based on some input (text). We are going to train the model based on a txt file, which [contains text from Shakespeare's work](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).


## Building on the foundation of Bigram Model.

Although I have just mentioned the shortcomings of n-gram models, to keep thins simple, we will start with a traditional **bigram model**, where we will predict the next character based only on the current one. You saw right: we will predict the next character. This is just for simplicity. In a real world scenario, we would predict the next token or next word, but for now, we will stick with characters.

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

Now that we have inputs, we can start to build our model. The first model will be very simple, it will just consist of an embedding layer.
The embedding layer will map the input characters to vectors. This vector will be the representation of a sequence of characters, it stores information on the characters in the sequence. It will be used during the training process: the model will learn to predict the next character based on this vector representation.

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
The code above is the implementation of the bigram model. It has two methods: the `forward` and the `generate` methods.
The `forward` method is used during the training process. It takes the input characters and the target characters as input and returns the `logits` and the `loss`.
The `logits` is the output of the model, it is the prediction of the next character. The `loss` is the difference between the prediction and the target.
We can use this information to update the weights of the model during the training process. (See example code in collapsabel details below.)

<details>

```python
for steps in range(eval_iters):
    if steps % print_iter == 0:
        losses = estimate_loss()
        print(f"step: {steps}, train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    # The implementation of the get_batch function is not in this blogpost, but it generates a batch of input and target sequences from either training or validation data by randomly sampling starting indices and extracting sequences of a specified length.
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

The `generate` method is used to generate text. It takes the input characters and the number of characters we want to generate as input and returns a character based on the probability distribution of the possible next characters.
It repeats this process until it generates the desired number of characters.


## Step 2: Introducing the Self-Attention Mechanism

Now you might be able to imagine, that this process is too simple to work well and it doesn't. For example, I got the following text: `WzPZ!
P hathoing ENCORDWhw.zhin lg ik m Whanthanewousove t; t fodeaMgCHxqXAullow's tScedhamas t p an`.

To overcome the limitations of the bigram model, we introduced **self-attention**, the cornerstone of transformers. Self-attention enables each token in the sequence to dynamically weigh the importance of preceeding tokens when making predictions.

### How Self-Attention Works
Attention - as the name suggests - is about focusing on the important parts of the input. For instance, in a Shakespearean text, the word “king” should attend to “queen” and “throne” with high weights, while ignoring unrelated words like “apple” or “car” or very general words like "the", because they don't have any relevance to the word "king" in the text and/or don't help to predict the next word.

In the context of transformers, self-attention involves computing a set of queries, keys, and values for each token in the sequence. Each have different roles:
* The query encapsulates what kind of information the token is looking for, when it attends other tokens.
* The key represents what kind of information the token has to share.
* The value contains information about the token.

I would like to share this great example (after reading [this Stackoverflow thread](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) - yeah, I know, so oldschool): imagine you are searching for a video on Youtube. In this context, the query is the search term you enter, the keys are the 'properties' of the of the videos (like title, description, etc.), and the values are the videos themselves. The attention mechanism helps you to find the most relevant videos (give back 'value' with highest probability) based on your search term ('query') and the properties ('keys') of the videos.

The same thing happens with the tokens/words in a sequence. The model assigns queries and keys to each token. Each token is compared to the keys of all other tokens, generating a set of scores that represent “relevance.” These 'relevance; scores are then used to compute a weighted sum of the values (weighting the tokens based on their relevance). Ideally, the most relevant tokens will have the highest weights, which helps the model to select the correct token in the sequence.

### The simple trick to calculate the attention scores

Maybe it is a bit of a detail, but I really found this highlight interesting in Andrej's video. As described above, the attention mechanism is about calculating the relevance of the tokens in the sequence.

So what does this mean? Let's say we have tokens represented as vectors.

token1 = [0.1, 0.2]

token2 = [0.3, 0.4]

token3 = [0.5, 0.6]

Then our tokenised sentence, represented as vectors would be:

sentence = [token1, token2, token3]
sentence = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

Now our task is to calculate the relevance of the previous tokens for each token. This means we have to calculate the relevance of token1 for token2 and the relevance of token1, token2 for token3. We actually just calculate the average of the previous tokens for each token. You could calculate this by iterating over the tokens and calculating the average of the previous tokens for each token:

step 1: average of token1 = token1
step 2: average of token2 = (token1 + token2) / 2
step 3: average of token3 = (token1 + token2 + token3) / 3

<details>
```python
sentence = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
for i, token in enumerate(sentence):
    previous_tokens = sentence[:i+1]
    relevance = torch.stack(list(previous_tokens)).mean(dim=0)
    print(relevance)
```
</details>

However, iterating over each token is not efficient. Usually we have a lot of tokens in a sequence and also the vector representation of a token is much longer.

With neural networks usually the trick is to use matrix multiplication, which makes calculations much more efficient. So how is it done in this case?



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

