# How Transformer Models Work: Breaking Down the Magic of GPT

## Introduction
I wanted to get a deeper understanding on how GPT models work. I have watched Andrej Karpaty's [amazing Youtube video](https://www.youtube.com/watch?v=kCc8FmEb1nY&t=2s) on this.
In this blogpost I will write what I have learned. My main goals with this blogpost are:
1. To check, if I have really understood the video,
2. To preserve this knowledge for my future self,
3. To share this knowledge with the world.

So how do transformer models, like those powering GPT, work? While the final models may seem like black boxes, their underlying mechanisms are surprisingly structured and elegant.

Of course I don't want to go as deep as Andrej does in his video, so if you want to get a real deep understanding, I really suggest to watch his video! I will just highlight the main steps and ellaborate on some parts, which I think are interesting.

I will also provide some code in the hope, that it helps understanding the concepts better. However, it is not a full implementation. If you want a full code, with all the dependencies to run it, I suggest to check out Andrej's github repository.

## Understanding the Problem: Context and Sequence in Language Modeling

Our goal at the end of the day is to generate text that help you achieve your task. Text generation is a huge topic, it can mean summarising, translating or having a discussion with a chatbot.

Transformer models are great for tasks, where tasks involve predicting the next word (or character) in a sequence based on prior context. For instance, given the phrase “To be or not to”, a good model should predict “be” as the next word. There are traditional models, like n-grams (eg.: `bigram`), but they struggle to capture long-range dependencies effectively. These models usually take the last n words/characters/tokens as an input and predict the next one based on all these inputs.

Transformers address this limitation by focusing on the relationship between all elements in a sequence through their self-attention mechanism.

In this blogpost I am going to use the same example Andrej uses in his video. Our task is to implement a model, which can generate text, based on some input (text). We are going to train the model based on a txt file, which [contains text from Shakespeare's work](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt).


## Predicting the next character - building on the foundation of Bigram Model.

Although I have just mentioned the shortcomings of n-gram models, to keep things simple, we can showcase with a traditional **bigram model**, where we will predict the next character based only on the current one. You saw right: we will predict the next character. This is just for simplicity. In a real world scenario, we would predict the next token or next word, but for now, we will stick with characters.

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


## Introducing the Self-Attention Mechanism

Now you might be able to imagine, that this process is too simple to work well and it doesn't. For example, I got the following text: `WzPZ!
P hathoing ENCORDWhw.zhin lg ik m Whanthanewousove t; t fodeaMgCHxqXAullow's tScedhamas t p an`.

To overcome the limitations of the bigram model, we introduced **self-attention**, the cornerstone of transformers. Self-attention enables each token in the sequence to dynamically weigh the importance of preceeding tokens when making predictions.

### How Self-Attention Works
Attention - as the name suggests - is about focusing on the important parts of the input. For instance, in a Shakespearean text, the word “king” should attend to “queen” and “throne” with high weights, while ignoring unrelated words like “apple” or “car” or very general words like "the", because they don't have any relevance to the word "king" in the text and/or don't help to predict the next word.

In the context of transformers, self-attention involves computing a set of queries, keys, and values for each token in the sequence. Each have different roles:
* The query encapsulates what kind of information the token is looking for, when it attends other tokens.
* The key represents what kind of information the token has to share.
* The value contains information about the token.

I would like to share this great example (after reading [this Stackoverflow thread](https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms) - yeah, I know, so oldschool to get information from SO): imagine you are searching for a video on Youtube. In this context, the query is the search term you enter, the keys are the 'properties' of the of the videos (like title, description, etc.), and the values are the videos themselves. The attention mechanism helps you to find the most relevant videos (give back 'value' with highest probability) based on your search term ('query') and the properties ('keys') of the videos.

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

This code above will calculate the average for previous tokens:
```
[0.1000, 0.2000]
[0.2000, 0.3000]
[0.3000, 0.4000]
```
The first token is doesn't have any tokens before, so we get back the same vector.
For the second token we calculate the average of the first and the second: [(0.1+0.3)/2, (0.2+0.4)/2]. For the third token we calculate the average of first, second and third.

However, iterating over each token is not efficient. Usually we have a lot of tokens in a sequence and also the vector representation of a token is much longer.

Computation in neural networks are usually done with matrix multiplication, because it makes calculations much more efficient. So how is it done in this case? The trick is to store the token representations in a matrix, but use a lower triangular matrix to calculate the relevance of the previous tokens for each token. A lower triangular matrix is a matrix, where all the elements above the diagonal are 0, eg.:
```
[1, 0, 0]
[1, 1, 0]
[1, 1, 1]
```

This is useful, because this way we can calculate the relevance of the previous tokens for each token with a single matrix multiplication.

First we need a lower triangular matrix, which will help us to calculate the average of the previous tokens. This can be done by giving the same weight to previous tokens (and givin 0 weight to future tokens).
```
[1.00, 0.00, 0.00]
[0.50, 0.50, 0.00]
[0.33, 0.33, 0.33]
```

Then we use this lower triangular matrix and multiply it with the matrix representation of our vectors:

```python
lower_triangular_matrix = torch.tril(torch.ones((3, 3)))
lower_triangular_matrix = lower_triangular_matrix / torch.sum(lower_triangular_matrix, 1, keepdim=True)

sentence = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

print(sentence @ lower_triangular_matrix)
```

Now in the actual calculation we do a bit more, instead of just using the average, but it doesn't change the message: using matrices makes calculation much more efficient!


## Putting it together - The Transformer Block

Up until now we have seen how to predict the next character based on the previous ones using a simple bigram model and how to use self-attention to calculate the relevance of the previous tokens for each token. Now we will put these concepts together and build a transformer block, which is the building block of transformer models like GPT.

A single **transformer block** integrates:

1. **Multi-head attention:** This extends self-attention by running multiple attention mechanisms in parallel. Each 'head' learns to focus on different aspects of the sequence—such as syntax, semantics, or specific recurring patterns. For our Shakespearean text example, one head might focus on rhythmic patterns (e.g., iambic pentameter), while another emphasizes grammatical structure. The goal is basically to analyise the text from many different perspectives to make the next token predictions as good as possible.
2. **Feedforward layers:** Applying non-linear transformations to learn more complex (non-linear) patterns in the data. One important step here is the *Dropout layer*. While this technique is good for finding complex patterns, it is prone to overfitting. The Dropout layer helps this by dropping the communication between some neurons. This 'dumbs down' the model during training to make sure it is not overfitting on the training data.
3. **Residual connections:** Adding the input to the output of each layer to stabilize gradients and preserve information. This is a technique, which is used to avoid the vanishing gradient problem, which is a problem, when the gradients are too small and the model is not able to learn the correct patterns.
4. **Layer normalization:** Ensuring numerical stability by normalizing inputs within each layer. Instability would mean that the gradients are too large or too small, which would make the training process unstable: the weights in the model would be too large or too small and the model would not be able to learn the correct patterns. (So it solves a similar problem as the Residual connections, just the root source of the problem is different, so a different solution is needed.)

By stacking multiple transformer blocks, the model iteratively refines its understanding of the input, enabling it to generate more coherent and contextually accurate text. The final architecture can look multiple ways, one common example can be found in the the ["Attention is All You Need" paper](https://arxiv.org/abs/1706.03762).

## Training the Model

The model is trained by:

1. Defining a loss function, that measures how well the predicted sequence matches the actual sequence. Usually cross-entropy is used for such classification tasks: it measures the difference between two probability distributions.
2. Using backpropagation to compute gradients of the loss with respect to model parameters.
3. Optimizing parameters using a gradient descent algorithm such as AdamW.

For our Shakespearean text, training involves exposing the model to sequences of characters and adjusting its parameters to minimize the prediction error. Over time, the model learns patterns like word structures and stylistic nuances.

## Conclusion

The example above shows a so called 'decoder' solution, which takes a sequence of tokens (characters in our case) and predicts the next token in the sequence. This is a common architecture for text generation tasks, where the model generates text based on a given input.

There are some differences on how models like GPT actually work.

For one, they use an encoder-decoder architecture. This is useful when you want to depend on a text, use that as an input and generate text based on that. For example, the [original paper](https://arxiv.org/abs/1706.03762) focused on translations. The input here was eg.: sentence in French. Based on this, we want to generate text in English.

In the decoder there is another addition, the cross-attention layer. Here the queries are still generated from x (input from text in decoder), but keys and values are coming from the encoder. This means that the decoding is not only conditiond on past text during generation, but also the full, embedded input text.

Other than the differences in the architecture, what else we would need if we would want to 'replicate' chatGPT? There we have a pretraining phase and a finetuning phase as well.

In pretraining we train on a large chunk of the internet and create a decoder-only transformer, so it can generate text. Much like we did in our code above. The difference is of course size. With the parameters Karpathy sets by the end of video (which could not run on a MacBook without GPU) we have ~10M parameters. Our dataset is ~1M characters. This would be ~300K tokens in OpenAI's vocabulary.

For OpenAI's GPT models GPT-3 has 175B parameters. They have trained it on 300B tokens. The code architecture is similar, the hardware is the difference, they use thousends of GPUs for training.

After pretraining, we don't have something, which can answer questions, it just completes documents, generates texts.
We need finetuning to make sure it behaves as we want, so it answers questions.
For this OpenAI has 3 steps:
1. Supervised learning with dataset gathered for this purpose (so questions with answers).
2. Humans look at different responses of the AI and rank them (We generate multiple anwers to a question. Then these answers are ranked.) They use it to train a reward model.
3. With the reward model they run PPO reinforcement learning algorithm. So when we have a question (a prompt), the AI should generate an answer which get's a high score from the rewar model.
This takes the model from a document generator to a question answerer.

So while we cannot build a full GPT, I hope this blogpost gives you a good understanding of how transformer models work and how they can be used for text generation tasks. The concepts of self-attention, multi-head attention, and the transformer block are key to understanding the power of these models.




