# Creating a Sentiment Analysis Web App
## Using PyTorch and SageMaker

_Deep Learning Nanodegree Program | Deployment_

---

Now that we have a basic understanding of how SageMaker works we will try to use it to construct a complete project from end to end. Our goal will be to have a simple web page which a user can use to enter a movie review. The web page will then send the review off to our deployed model which will predict the sentiment of the entered review.

## Instructions

Some template code has already been provided for you, and you will need to implement additional functionality to successfully complete this notebook. You will not need to modify the included code beyond what is requested. Sections that begin with '**TODO**' in the header indicate that you need to complete or implement some portion within them. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `# TODO: ...` comment. Please be sure to read the instructions carefully!

In addition to implementing code, there will be questions for you to answer which relate to the task and your implementation. Each section where you will answer a question is preceded by a '**Question:**' header. Carefully read each question and provide your answer below the '**Answer:**' header by editing the Markdown cell.

> **Note**: Code and Markdown cells can be executed using the **Shift+Enter** keyboard shortcut. In addition, a cell can be edited by typically clicking it (double-click for Markdown cells) or by pressing **Enter** while it is highlighted.

## General Outline

Recall the general outline for SageMaker projects using a notebook instance.

1. Download or otherwise retrieve the data.
2. Process / Prepare the data.
3. Upload the processed data to S3.
4. Train a chosen model.
5. Test the trained model (typically using a batch transform job).
6. Deploy the trained model.
7. Use the deployed model.

For this project, you will be following the steps in the general outline with some modifications. 

First, you will not be testing the model in its own step. You will still be testing the model, however, you will do it by deploying your model and then using the deployed model by sending the test data to it. One of the reasons for doing this is so that you can make sure that your deployed model is working correctly before moving forward.

In addition, you will deploy and use your trained model a second time. In the second iteration you will customize the way that your trained model is deployed by including some of your own code. In addition, your newly deployed model will be used in the sentiment analysis web app.

## Step 1: Downloading the data

As in the XGBoost in SageMaker notebook, we will be using the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/)

> Maas, Andrew L., et al. [Learning Word Vectors for Sentiment Analysis](http://ai.stanford.edu/~amaas/data/sentiment/). In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_. Association for Computational Linguistics, 2011.


```python
%mkdir ../data
!wget -O ../data/aclImdb_v1.tar.gz http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -zxf ../data/aclImdb_v1.tar.gz -C ../data
```

## Step 2: Preparing and Processing the data

Also, as in the XGBoost notebook, we will be doing some initial data processing. The first few steps are the same as in the XGBoost example. To begin with, we will read in each of the reviews and combine them into a single input structure. Then, we will split the dataset into a training set and a testing set.


```python
import os
import glob

def read_imdb_data(data_dir='../data/aclImdb'):
    data = {}
    labels = {}
    
    for data_type in ['train', 'test']:
        data[data_type] = {}
        labels[data_type] = {}
        
        for sentiment in ['pos', 'neg']:
            data[data_type][sentiment] = []
            labels[data_type][sentiment] = []
            
            path = os.path.join(data_dir, data_type, sentiment, '*.txt')
            files = glob.glob(path)
            
            for f in files:
                with open(f) as review:
                    data[data_type][sentiment].append(review.read())
                    # Here we represent a positive review by '1' and a negative review by '0'
                    labels[data_type][sentiment].append(1 if sentiment == 'pos' else 0)
                    
            assert len(data[data_type][sentiment]) == len(labels[data_type][sentiment]), \
                    "{}/{} data size does not match labels size".format(data_type, sentiment)
                
    return data, labels
```

IMDB reviews: train = 12500 pos / 12500 neg, test = 12500 pos / 12500 neg

IMDb reviews (combined): train = 25000, test = 25000


Now that we have our training and testing sets unified and prepared, we should do a quick check and see an example of the data our model will be trained on. This is generally a good idea as it allows you to see how each of the further processing steps affects the reviews and it also ensures that the data has been loaded correctly.

Example:
    
I was very excited about seeing this film, anticipating a visual excursus on the relation of artistic beauty and nature, containing the kinds of wisdom the likes of "Rivers and Tides." However, that's not what I received. Instead, I get a fairly uninspired film about how human industry is bad for nature. Which is clearly a quite unorthodox claim.<br /><br />The photographer seems conflicted about the aesthetic qualities of his images and the supposed "ethical" duty he has to the workers occasionally peopling the images, along the periphery. And frankly, the images were not generally that impressive. And according to this "artist," scale is the basis for what makes something beautiful.<br /><br />In all respects, a stupid film. For people who'd like to feel better about their environmental consciousness ... but not for any one who would like to think about the complexities of the issues surrounding it.
    
0
    

The `review_to_words` method defined above uses `BeautifulSoup` to remove any html tags that appear and uses the `nltk` package to tokenize the reviews. As a check to ensure we know how everything is working, try applying `review_to_words` to one of the reviews in the training set.

['excit',
 'see',
 'film',
 'anticip',
 'visual',
 'excursu',
 'relat',
 'artist',
 'beauti',
 'natur',
 'contain',
 'kind',
 'wisdom',
 'like',
 'river',
 'tide',
 'howev',
 'receiv',
 'instead',
 'get',
 'fairli',
 'uninspir',
 'film',
 'human',
 'industri',
 'bad',
 'natur',
 'clearli',
 'quit',
 'unorthodox',
 'claim',
 'photograph',
 'seem',
 'conflict',
 'aesthet',
 'qualiti',
 'imag',
 'suppos',
 'ethic',
 'duti',
 'worker',
 'occasion',
 'peopl',
 'imag',
 'along',
 'peripheri',
 'frankli',
 'imag',
 'gener',
 'impress',
 'accord',
 'artist',
 'scale',
 'basi',
 'make',
 'someth',
 'beauti',
 'respect',
 'stupid',
 'film',
 'peopl',
 'like',
 'feel',
 'better',
 'environment',
 'conscious',
 'one',
 'would',
 'like',
 'think',
 'complex',
 'issu',
 'surround']

**Question:** Above we mentioned that `review_to_words` method removes html formatting and allows us to tokenize the words found in a review, for example, converting *entertained* and *entertaining* into *entertain* so that they are treated as though they are the same word. What else, if anything, does this method do to the input?

**Answer:**

The method below applies the `review_to_words` method to each of the reviews in the training and testing datasets. In addition it caches the results. This is because performing this processing step can take a long time. This way if you are unable to complete the notebook in the current session, you can come back without needing to process the data a second time.


```python
import pickle

cache_dir = os.path.join("../cache", "sentiment_analysis")  # where to store cache files
os.makedirs(cache_dir, exist_ok=True)  # ensure cache directory exists

def preprocess_data(data_train, data_test, labels_train, labels_test,
                    cache_dir=cache_dir, cache_file="preprocessed_data.pkl"):
    """Convert each review to words; read from cache if available."""

    # If cache_file is not None, try to read from it first
    cache_data = None
    if cache_file is not None:
        try:
            with open(os.path.join(cache_dir, cache_file), "rb") as f:
                cache_data = pickle.load(f)
            print("Read preprocessed data from cache file:", cache_file)
        except:
            pass  # unable to read from cache, but that's okay
    
    # If cache is missing, then do the heavy lifting
    if cache_data is None:
        # Preprocess training and test data to obtain words for each review
        #words_train = list(map(review_to_words, data_train))
        #words_test = list(map(review_to_words, data_test))
        words_train = [review_to_words(review) for review in data_train]
        words_test = [review_to_words(review) for review in data_test]
        
        # Write to cache file for future runs
        if cache_file is not None:
            cache_data = dict(words_train=words_train, words_test=words_test,
                              labels_train=labels_train, labels_test=labels_test)
            with open(os.path.join(cache_dir, cache_file), "wb") as f:
                pickle.dump(cache_data, f)
            print("Wrote preprocessed data to cache file:", cache_file)
    else:
        # Unpack data loaded from cache file
        words_train, words_test, labels_train, labels_test = (cache_data['words_train'],
                cache_data['words_test'], cache_data['labels_train'], cache_data['labels_test'])
    
    return words_train, words_test, labels_train, labels_test
```

# Preprocess data
train_X, test_X, train_y, test_y = preprocess_data(train_X, test_X, train_y, test_y)

## Transform the data

In the XGBoost notebook we transformed the data from its word representation to a bag-of-words feature representation. For the model we are going to construct in this notebook we will construct a feature representation which is very similar. To start, we will represent each word as an integer. Of course, some of the words that appear in the reviews occur very infrequently and so likely don't contain much information for the purposes of sentiment analysis. The way we will deal with this problem is that we will fix the size of our working vocabulary and we will only include the words that appear most frequently. We will then combine all of the infrequent words into a single category and, in our case, we will label it as `1`.

Since we will be using a recurrent neural network, it will be convenient if the length of each review is the same. To do this, we will fix a size for our reviews and then pad short reviews with the category 'no word' (which we will label `0`) and truncate long reviews.

### (TODO) Create a word dictionary

To begin with, we need to construct a way to map words that appear in the reviews to integers. Here we fix the size of our vocabulary (including the 'no word' and 'infrequent' categories) to be `5000` but you may wish to change this to see how it affects the model.

> **TODO:** Complete the implementation for the `build_dict()` method below. Note that even though the vocab_size is set to `5000`, we only want to construct a mapping for the most frequently appearing `4998` words. This is because we want to reserve the special labels `0` for 'no word' and `1` for 'infrequent word'.


```python
import numpy as np
from collections import Counter

def build_dict(data, vocab_size = 5000):
    """Construct and return a dictionary mapping each of the most frequently appearing words to a unique integer."""
    
    # TODO: Determine how often each word appears in `data`. Note that `data` is a list of sentences and that a
    #       sentence is a list of words.
    word_count = {} # A dict storing the words that appear in the reviews along with how often they occur
    word_count = Counter(np.concatenate(data, axis=0 ))# A dict storing the words that appear in the reviews along with how often they occur
    
    # TODO: Sort the words found in `data` so that sorted_words[0] is the most frequently appearing word and
    #       sorted_words[-1] is the least frequently appearing word.
    
    sorted_words = sorted(word_count, key=word_count.get, reverse=True)
    
    word_dict = {} # This is what we are building, a dictionary that translates words into integers
    for idx, word in enumerate(sorted_words[:vocab_size - 2]): # The -2 is so that we save room for the 'no word'
        word_dict[word] = idx + 2                              # 'infrequent' labels
        
    return word_dict
```

**Question:** What are the five most frequently appearing (tokenized) words in the training set? Does it makes sense that these words appear frequently in the training set?

**Answer:**

('movi', 51695),('film', 48190),('one', 27741),('like', 22799),('time', 16191) yeah for me it makes a lot of sense this words to be included frequently in a review I dont think this is exclusive of training set probably in the test dataset will be the same words

### Save `word_dict`

Later on when we construct an endpoint which processes a submitted review we will need to make use of the `word_dict` which we have created. As such, we will save it to a file now for future use.


```python
data_dir = '../data/pytorch' # The folder we will use for storing data
if not os.path.exists(data_dir): # Make sure that the folder exists
    os.makedirs(data_dir)
    
with open(os.path.join(data_dir, 'word_dict.pkl'), "wb") as f:
    pickle.dump(word_dict, f)
```

### Transform the reviews

Now that we have our word dictionary which allows us to transform the words appearing in the reviews into integers, it is time to make use of it and convert our reviews to their integer sequence representation, making sure to pad or truncate to a fixed length, which in our case is `500`.


```python
def convert_and_pad(word_dict, sentence, pad=500):
    NOWORD = 0 # We will use 0 to represent the 'no word' category
    INFREQ = 1 # and we use 1 to represent the infrequent words, i.e., words not appearing in word_dict
    
    working_sentence = [NOWORD] * pad
    
    for word_index, word in enumerate(sentence[:pad]):
        if word in word_dict:
            working_sentence[word_index] = word_dict[word]
        else:
            working_sentence[word_index] = INFREQ
            
    return working_sentence, min(len(sentence), pad)

def convert_and_pad_data(word_dict, data, pad=500):
    result = []
    lengths = []
    
    for sentence in data:
        converted, leng = convert_and_pad(word_dict, sentence, pad)
        result.append(converted)
        lengths.append(leng)
        
    return np.array(result), np.array(lengths)
```


```python
train_X, train_X_len = convert_and_pad_data(word_dict, train_X)
test_X, test_X_len = convert_and_pad_data(word_dict, test_X)
```

As a quick check to make sure that things are working as intended, check to see what one of the reviews in the training set looks like after having been processeed. Does this look reasonable? What is the length of a review in the training set?


```python
# Use this cell to examine one of the processed reviews to make sure everything is working as intended.
print(train_X[1])
print(train_X_len[0:1])
```

    [   1 2936   76  132  865  238  318  745 1007 1083 1420  132   25  319
        1  936 1330  396 2432    7   32 2936  107 1814    1  144   52   21
      407   83    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0    0    0    0    0
        0    0    0    0    0    0    0    0    0    0]
    [81]


**Question:** In the cells above we use the `preprocess_data` and `convert_and_pad_data` methods to process both the training and testing set. Why or why not might this be a problem?

**Answer:**

- preprocess_data = is a solution to store the data in case we need to retrieve it later, it will save time and processing if we close the notebook or lost conection

- convert_and_pad_data = it ensure to  maintain the data lenght constant

### Step 3: Upload the data to S3

As in the XGBoost notebook, we will need to upload the training dataset to S3 in order for our training code to access it. For now we will save it locally and we will upload to S3 later on.

### Save the processed training dataset locally

It is important to note the format of the data that we are saving as we will need to know it when we write the training code. In our case, each row of the dataset has the form label, length, review[500] where review[500] is a sequence of 500 integers representing the words in the review.

### Uploading the training data


Next, we need to upload the training data to the SageMaker default S3 bucket so that we can provide access to it while training our model.


```python
import sagemaker

sagemaker_session = sagemaker.Session()

bucket = sagemaker_session.default_bucket()
prefix = 'sagemaker/sentiment_rnn'

role = sagemaker.get_execution_role()
```


```python
input_data = sagemaker_session.upload_data(path=data_dir, bucket=bucket, key_prefix=prefix)
```

**NOTE:** The cell above uploads the entire contents of our data directory. This includes the `word_dict.pkl` file. This is fortunate as we will need this later on when we create an endpoint that accepts an arbitrary review. For now, we will just take note of the fact that it resides in the data directory (and so also in the S3 training bucket) and that we will need to make sure it gets saved in the model directory.

## Step 4: Build and Train the PyTorch Model

In the XGBoost notebook we discussed what a model is in the SageMaker framework. In particular, a model comprises three objects

 - Model Artifacts,
 - Training Code, and
 - Inference Code,
 
each of which interact with one another. In the XGBoost example we used training and inference code that was provided by Amazon. Here we will still be using containers provided by Amazon with the added benefit of being able to include our own custom code.

We will start by implementing our own neural network in PyTorch along with a training script. For the purposes of this project we have provided the necessary model object in the `model.py` file, inside of the `train` folder. You can see the provided implementation by running the cell below.


```python
!pygmentize train/model.py
```

    [34mimport[39;49;00m [04m[36mtorch.nn[39;49;00m [34mas[39;49;00m [04m[36mnn[39;49;00m
    
    [34mclass[39;49;00m [04m[32mLSTMClassifier[39;49;00m(nn.Module):
        [33m"""[39;49;00m
    [33m    This is the simple RNN model we will be using to perform Sentiment Analysis.[39;49;00m
    [33m    """[39;49;00m
    
        [34mdef[39;49;00m [32m__init__[39;49;00m([36mself[39;49;00m, embedding_dim, hidden_dim, vocab_size):
            [33m"""[39;49;00m
    [33m        Initialize the model by settingg up the various layers.[39;49;00m
    [33m        """[39;49;00m
            [36msuper[39;49;00m(LSTMClassifier, [36mself[39;49;00m).[32m__init__[39;49;00m()
    
            [36mself[39;49;00m.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=[34m0[39;49;00m)
            [36mself[39;49;00m.lstm = nn.LSTM(embedding_dim, hidden_dim)
            [36mself[39;49;00m.dense = nn.Linear(in_features=hidden_dim, out_features=[34m1[39;49;00m)
            [36mself[39;49;00m.sig = nn.Sigmoid()
            
            [36mself[39;49;00m.word_dict = [36mNone[39;49;00m
    
        [34mdef[39;49;00m [32mforward[39;49;00m([36mself[39;49;00m, x):
            [33m"""[39;49;00m
    [33m        Perform a forward pass of our model on some input.[39;49;00m
    [33m        """[39;49;00m
            x = x.t()
            lengths = x[[34m0[39;49;00m,:]
            reviews = x[[34m1[39;49;00m:,:]
            embeds = [36mself[39;49;00m.embedding(reviews)
            lstm_out, _ = [36mself[39;49;00m.lstm(embeds)
            out = [36mself[39;49;00m.dense(lstm_out)
            out = out[lengths - [34m1[39;49;00m, [36mrange[39;49;00m([36mlen[39;49;00m(lengths))]
            [34mreturn[39;49;00m [36mself[39;49;00m.sig(out.squeeze())


The important takeaway from the implementation provided is that there are three parameters that we may wish to tweak to improve the performance of our model. These are the embedding dimension, the hidden dimension and the size of the vocabulary. We will likely want to make these parameters configurable in the training script so that if we wish to modify them we do not need to modify the script itself. We will see how to do this later on. To start we will write some of the training code in the notebook so that we can more easily diagnose any issues that arise.

First we will load a small portion of the training data set to use as a sample. It would be very time consuming to try and train the model completely in the notebook as we do not have access to a gpu and the compute instance that we are using is not particularly powerful. However, we can work on a small bit of the data to get a feel for how our training script is behaving.


```python
import torch
import torch.utils.data
import pandas as pd

# Read in only the first 250 rows
train_sample = pd.read_csv(os.path.join(data_dir, 'train.csv'), header=None, names=None, nrows=250)

# Turn the input pandas dataframe into tensors
train_sample_y = torch.from_numpy(train_sample[[0]].values).float().squeeze()
train_sample_X = torch.from_numpy(train_sample.drop([0], axis=1).values).long()

# Build the dataset
train_sample_ds = torch.utils.data.TensorDataset(train_sample_X, train_sample_y)
# Build the dataloader
train_sample_dl = torch.utils.data.DataLoader(train_sample_ds, batch_size=50)
```

### (TODO) Writing the training method

Next we need to write the training code itself. This should be very similar to training methods that you have written before to train PyTorch models. We will leave any difficult aspects such as model saving / loading and parameter loading until a little later.


```python
def train(model, train_loader, epochs, optimizer, loss_fn, device):
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:         
            batch_X, batch_y = batch
            
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            # TODO: Complete this train method to train the model provided.
            optimizer.zero_grad()
            output = model.forward(batch_X)
            loss = loss_fn(output,batch_y)
            loss.backward()
            optimizer.step()
            
            
            total_loss += loss.data.item()
        print("Epoch: {}, BCELoss: {}".format(epoch, total_loss / len(train_loader)))
```

Supposing we have the training method above, we will test that it is working by writing a bit of code in the notebook that executes our training method on the small sample training set that we loaded earlier. The reason for doing this in the notebook is so that we have an opportunity to fix any errors that arise early when they are easier to diagnose.


```python
import torch.optim as optim
from train.model import LSTMClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(32, 100, 5000).to(device)
optimizer = optim.Adam(model.parameters())
loss_fn = torch.nn.BCELoss()

train(model, train_sample_dl, 5, optimizer, loss_fn, device)
```

    Epoch: 1, BCELoss: 0.6927193760871887
    Epoch: 2, BCELoss: 0.6816266179084778
    Epoch: 3, BCELoss: 0.6723000645637512
    Epoch: 4, BCELoss: 0.6620966792106628
    Epoch: 5, BCELoss: 0.6497733473777771


In order to construct a PyTorch model using SageMaker we must provide SageMaker with a training script. We may optionally include a directory which will be copied to the container and from which our training code will be run. When the training container is executed it will check the uploaded directory (if there is one) for a `requirements.txt` file and install any required Python libraries, after which the training script will be run.

### (TODO) Training the model

When a PyTorch model is constructed in SageMaker, an entry point must be specified. This is the Python file which will be executed when the model is trained. Inside of the `train` directory is a file called `train.py` which has been provided and which contains most of the necessary code to train our model. The only thing that is missing is the implementation of the `train()` method which you wrote earlier in this notebook.

**TODO**: Copy the `train()` method written above and paste it into the `train/train.py` file where required.

The way that SageMaker passes hyperparameters to the training script is by way of arguments. These arguments can then be parsed and used in the training script. To see how this is done take a look at the provided `train/train.py` file.


```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(entry_point="train.py",
                    source_dir="train",
                    role=role,
                    framework_version='0.4.0',
                    train_instance_count=1,
                    train_instance_type='ml.p2.xlarge',
                    hyperparameters={
                        'epochs': 10,
                        'hidden_dim': 200,
                    })
```


```python
estimator.fit({'training': input_data})
```

    2020-04-21 23:50:57 Starting - Starting the training job...
    2020-04-21 23:50:59 Starting - Launching requested ML instances...
    2020-04-21 23:51:55 Starting - Preparing the instances for training.........
    2020-04-21 23:53:11 Downloading - Downloading input data...
    2020-04-21 23:53:45 Training - Downloading the training image...
    2020-04-21 23:54:14 Training - Training image download completed. Training in progress.[34mbash: cannot set terminal process group (-1): Inappropriate ioctl for device[0m
    [34mbash: no job control in this shell[0m
    [34m2020-04-21 23:54:15,124 sagemaker-containers INFO     Imported framework sagemaker_pytorch_container.training[0m
    [34m2020-04-21 23:54:15,149 sagemaker_pytorch_container.training INFO     Block until all host DNS lookups succeed.[0m
    [34m2020-04-21 23:54:18,175 sagemaker_pytorch_container.training INFO     Invoking user training script.[0m
    [34m2020-04-21 23:54:18,483 sagemaker-containers INFO     Module train does not provide a setup.py. [0m
    [34mGenerating setup.py[0m
    [34m2020-04-21 23:54:18,483 sagemaker-containers INFO     Generating setup.cfg[0m
    [34m2020-04-21 23:54:18,484 sagemaker-containers INFO     Generating MANIFEST.in[0m
    [34m2020-04-21 23:54:18,484 sagemaker-containers INFO     Installing module with the following command:[0m
    [34m/usr/bin/python -m pip install -U . -r requirements.txt[0m
    [34mProcessing /opt/ml/code[0m
    [34mCollecting pandas (from -r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/74/24/0cdbf8907e1e3bc5a8da03345c23cbed7044330bb8f73bb12e711a640a00/pandas-0.24.2-cp35-cp35m-manylinux1_x86_64.whl (10.0MB)[0m
    [34mCollecting numpy (from -r requirements.txt (line 2))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/45/25/48e4ea892e93348d48a3a0d23ad94b176d6ab66084efcd881c78771d4abf/numpy-1.18.3-cp35-cp35m-manylinux1_x86_64.whl (20.0MB)[0m
    [34mCollecting nltk (from -r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/92/75/ce35194d8e3022203cca0d2f896dbb88689f9b3fce8e9f9cff942913519d/nltk-3.5.zip (1.4MB)[0m
    [34mCollecting beautifulsoup4 (from -r requirements.txt (line 4))
      Downloading https://files.pythonhosted.org/packages/e8/b5/7bb03a696f2c9b7af792a8f51b82974e51c268f15e925fc834876a4efa0b/beautifulsoup4-4.9.0-py3-none-any.whl (109kB)[0m
    [34mCollecting html5lib (from -r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/a5/62/bbd2be0e7943ec8504b517e62bab011b4946e1258842bc159e5dfde15b96/html5lib-1.0.1-py2.py3-none-any.whl (117kB)[0m
    [34mRequirement already satisfied, skipping upgrade: python-dateutil>=2.5.0 in /usr/local/lib/python3.5/dist-packages (from pandas->-r requirements.txt (line 1)) (2.7.5)[0m
    [34mCollecting pytz>=2011k (from pandas->-r requirements.txt (line 1))
      Downloading https://files.pythonhosted.org/packages/e7/f9/f0b53f88060247251bf481fa6ea62cd0d25bf1b11a87888e53ce5b7c8ad2/pytz-2019.3-py2.py3-none-any.whl (509kB)[0m
    [34mRequirement already satisfied, skipping upgrade: click in /usr/local/lib/python3.5/dist-packages (from nltk->-r requirements.txt (line 3)) (7.0)[0m
    [34mCollecting joblib (from nltk->-r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/28/5c/cf6a2b65a321c4a209efcdf64c2689efae2cb62661f8f6f4bb28547cf1bf/joblib-0.14.1-py2.py3-none-any.whl (294kB)[0m
    [34mCollecting regex (from nltk->-r requirements.txt (line 3))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/4c/e7/eee73c42c1193fecc0e91361a163cbb8dfbea62c3db7618ad986e5b43a14/regex-2020.4.4.tar.gz (695kB)[0m
    [34mCollecting tqdm (from nltk->-r requirements.txt (line 3))
      Downloading https://files.pythonhosted.org/packages/4a/1c/6359be64e8301b84160f6f6f7936bbfaaa5e9a4eab6cbc681db07600b949/tqdm-4.45.0-py2.py3-none-any.whl (60kB)[0m
    [34mCollecting soupsieve>1.2 (from beautifulsoup4->-r requirements.txt (line 4))[0m
    [34m  Downloading https://files.pythonhosted.org/packages/05/cf/ea245e52f55823f19992447b008bcbb7f78efc5960d77f6c34b5b45b36dd/soupsieve-2.0-py2.py3-none-any.whl[0m
    [34mRequirement already satisfied, skipping upgrade: six>=1.9 in /usr/local/lib/python3.5/dist-packages (from html5lib->-r requirements.txt (line 5)) (1.11.0)[0m
    [34mCollecting webencodings (from html5lib->-r requirements.txt (line 5))
      Downloading https://files.pythonhosted.org/packages/f4/24/2a3e3df732393fed8b3ebf2ec078f05546de641fe1b667ee316ec1dcf3b7/webencodings-0.5.1-py2.py3-none-any.whl[0m
    [34mBuilding wheels for collected packages: nltk, train, regex
      Running setup.py bdist_wheel for nltk: started[0m
    [34m  Running setup.py bdist_wheel for nltk: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/ae/8c/3f/b1fe0ba04555b08b57ab52ab7f86023639a526d8bc8d384306
      Running setup.py bdist_wheel for train: started
      Running setup.py bdist_wheel for train: finished with status 'done'
      Stored in directory: /tmp/pip-ephem-wheel-cache-h2_11nn4/wheels/35/24/16/37574d11bf9bde50616c67372a334f94fa8356bc7164af8ca3
      Running setup.py bdist_wheel for regex: started[0m
    [34m  Running setup.py bdist_wheel for regex: finished with status 'done'
      Stored in directory: /root/.cache/pip/wheels/e6/9b/ae/2972da29cc7759b71dee015813b7c6931917d6a51e64ed5e79[0m
    [34mSuccessfully built nltk train regex[0m
    [34mInstalling collected packages: numpy, pytz, pandas, joblib, regex, tqdm, nltk, soupsieve, beautifulsoup4, webencodings, html5lib, train
      Found existing installation: numpy 1.15.4
        Uninstalling numpy-1.15.4:[0m
    [34m      Successfully uninstalled numpy-1.15.4[0m
    [34mSuccessfully installed beautifulsoup4-4.9.0 html5lib-1.0.1 joblib-0.14.1 nltk-3.5 numpy-1.18.3 pandas-0.24.2 pytz-2019.3 regex-2020.4.4 soupsieve-2.0 tqdm-4.45.0 train-1.0.0 webencodings-0.5.1[0m
    [34mYou are using pip version 18.1, however version 20.1b1 is available.[0m
    [34mYou should consider upgrading via the 'pip install --upgrade pip' command.[0m
    [34m2020-04-21 23:54:41,131 sagemaker-containers INFO     Invoking user script
    [0m
    [34mTraining Env:
    [0m
    [34m{
        "log_level": 20,
        "current_host": "algo-1",
        "resource_config": {
            "current_host": "algo-1",
            "hosts": [
                "algo-1"
            ],
            "network_interface_name": "eth0"
        },
        "channel_input_dirs": {
            "training": "/opt/ml/input/data/training"
        },
        "module_name": "train",
        "framework_module": "sagemaker_pytorch_container.training:main",
        "input_config_dir": "/opt/ml/input/config",
        "additional_framework_parameters": {},
        "job_name": "sagemaker-pytorch-2020-04-21-23-50-57-492",
        "output_data_dir": "/opt/ml/output/data",
        "input_dir": "/opt/ml/input",
        "module_dir": "s3://sagemaker-us-east-2-927442490306/sagemaker-pytorch-2020-04-21-23-50-57-492/source/sourcedir.tar.gz",
        "user_entry_point": "train.py",
        "model_dir": "/opt/ml/model",
        "output_dir": "/opt/ml/output",
        "input_data_config": {
            "training": {
                "S3DistributionType": "FullyReplicated",
                "RecordWrapperType": "None",
                "TrainingInputMode": "File"
            }
        },
        "hyperparameters": {
            "epochs": 10,
            "hidden_dim": 200
        },
        "output_intermediate_dir": "/opt/ml/output/intermediate",
        "num_gpus": 1,
        "hosts": [
            "algo-1"
        ],
        "network_interface_name": "eth0",
        "num_cpus": 4[0m
    [34m}
    [0m
    [34mEnvironment variables:
    [0m
    [34mSM_OUTPUT_DIR=/opt/ml/output[0m
    [34mSM_NUM_GPUS=1[0m
    [34mSM_MODULE_NAME=train[0m
    [34mSM_CHANNEL_TRAINING=/opt/ml/input/data/training[0m
    [34mSM_OUTPUT_INTERMEDIATE_DIR=/opt/ml/output/intermediate[0m
    [34mSM_INPUT_DATA_CONFIG={"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}}[0m
    [34mPYTHONPATH=/usr/local/bin:/usr/lib/python35.zip:/usr/lib/python3.5:/usr/lib/python3.5/plat-x86_64-linux-gnu:/usr/lib/python3.5/lib-dynload:/usr/local/lib/python3.5/dist-packages:/usr/lib/python3/dist-packages[0m
    [34mSM_HOSTS=["algo-1"][0m
    [34mSM_NUM_CPUS=4[0m
    [34mSM_FRAMEWORK_MODULE=sagemaker_pytorch_container.training:main[0m
    [34mSM_RESOURCE_CONFIG={"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"}[0m
    [34mSM_HP_EPOCHS=10[0m
    [34mSM_OUTPUT_DATA_DIR=/opt/ml/output/data[0m
    [34mSM_MODEL_DIR=/opt/ml/model[0m
    [34mSM_CURRENT_HOST=algo-1[0m
    [34mSM_LOG_LEVEL=20[0m
    [34mSM_MODULE_DIR=s3://sagemaker-us-east-2-927442490306/sagemaker-pytorch-2020-04-21-23-50-57-492/source/sourcedir.tar.gz[0m
    [34mSM_CHANNELS=["training"][0m
    [34mSM_FRAMEWORK_PARAMS={}[0m
    [34mSM_USER_ARGS=["--epochs","10","--hidden_dim","200"][0m
    [34mSM_INPUT_DIR=/opt/ml/input[0m
    [34mSM_HPS={"epochs":10,"hidden_dim":200}[0m
    [34mSM_HP_HIDDEN_DIM=200[0m
    [34mSM_TRAINING_ENV={"additional_framework_parameters":{},"channel_input_dirs":{"training":"/opt/ml/input/data/training"},"current_host":"algo-1","framework_module":"sagemaker_pytorch_container.training:main","hosts":["algo-1"],"hyperparameters":{"epochs":10,"hidden_dim":200},"input_config_dir":"/opt/ml/input/config","input_data_config":{"training":{"RecordWrapperType":"None","S3DistributionType":"FullyReplicated","TrainingInputMode":"File"}},"input_dir":"/opt/ml/input","job_name":"sagemaker-pytorch-2020-04-21-23-50-57-492","log_level":20,"model_dir":"/opt/ml/model","module_dir":"s3://sagemaker-us-east-2-927442490306/sagemaker-pytorch-2020-04-21-23-50-57-492/source/sourcedir.tar.gz","module_name":"train","network_interface_name":"eth0","num_cpus":4,"num_gpus":1,"output_data_dir":"/opt/ml/output/data","output_dir":"/opt/ml/output","output_intermediate_dir":"/opt/ml/output/intermediate","resource_config":{"current_host":"algo-1","hosts":["algo-1"],"network_interface_name":"eth0"},"user_entry_point":"train.py"}[0m
    [34mSM_USER_ENTRY_POINT=train.py[0m
    [34mSM_INPUT_CONFIG_DIR=/opt/ml/input/config[0m
    [34mSM_NETWORK_INTERFACE_NAME=eth0
    [0m
    [34mInvoking script with the following command:
    [0m
    [34m/usr/bin/python -m train --epochs 10 --hidden_dim 200
    
    [0m
    [34mUsing device cuda.[0m
    [34mGet train data loader.[0m
    [34mModel loaded with embedding_dim 32, hidden_dim 200, vocab_size 5000.[0m
    [34mEpoch: 1, BCELoss: 0.6675110799925668[0m
    [34mEpoch: 2, BCELoss: 0.5918845716787844[0m
    [34mEpoch: 3, BCELoss: 0.5081290590519808[0m
    [34mEpoch: 4, BCELoss: 0.4426859781450155[0m
    [34mEpoch: 5, BCELoss: 0.39109693862953965[0m
    [34mEpoch: 6, BCELoss: 0.3640510871702311[0m
    [34mEpoch: 7, BCELoss: 0.36082656651127093[0m
    [34mEpoch: 8, BCELoss: 0.36352434936834843[0m
    [34mEpoch: 9, BCELoss: 0.3003002733600383[0m
    [34mEpoch: 10, BCELoss: 0.2762880118525758[0m
    [34m2020-04-21 23:57:38,036 sagemaker-containers INFO     Reporting training SUCCESS[0m
    
    2020-04-21 23:57:48 Uploading - Uploading generated training model
    2020-04-21 23:57:48 Completed - Training job completed
    Training seconds: 277
    Billable seconds: 277


## Step 5: Testing the model

As mentioned at the top of this notebook, we will be testing this model by first deploying it and then sending the testing data to the deployed endpoint. We will do this so that we can make sure that the deployed model is working correctly.

## Step 6: Deploy the model for testing

Now that we have trained our model, we would like to test it to see how it performs. Currently our model takes input of the form `review_length, review[500]` where `review[500]` is a sequence of `500` integers which describe the words present in the review, encoded using `word_dict`. Fortunately for us, SageMaker provides built-in inference code for models with simple inputs such as this.

There is one thing that we need to provide, however, and that is a function which loads the saved model. This function must be called `model_fn()` and takes as its only parameter a path to the directory where the model artifacts are stored. This function must also be present in the python file which we specified as the entry point. In our case the model loading function has been provided and so no changes need to be made.

**NOTE**: When the built-in inference code is run it must import the `model_fn()` method from the `train.py` file. This is why the training code is wrapped in a main guard ( ie, `if __name__ == '__main__':` )

Since we don't need to change anything in the code that was uploaded during training, we can simply deploy the current model as-is.

**NOTE:** When deploying a model you are asking SageMaker to launch an compute instance that will wait for data to be sent to it. As a result, this compute instance will continue to run until *you* shut it down. This is important to know since the cost of a deployed endpoint depends on how long it has been running for.

In other words **If you are no longer using a deployed endpoint, shut it down!**

**TODO:** Deploy the trained model.


```python
# TODO: Deploy the trained model
predictor = estimator.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    ---------------!

## Step 7 - Use the model for testing

Once deployed, we can read in the test data and send it off to our deployed model to get some results. Once we collect all of the results we can determine how accurate our model is.


```python
test_X = pd.concat([pd.DataFrame(test_X_len), pd.DataFrame(test_X)], axis=1)
```


```python
test_X
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>67</td>
      <td>48</td>
      <td>528</td>
      <td>13</td>
      <td>2981</td>
      <td>1094</td>
      <td>1</td>
      <td>122</td>
      <td>38</td>
      <td>1494</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>93</td>
      <td>1</td>
      <td>1</td>
      <td>3413</td>
      <td>813</td>
      <td>97</td>
      <td>1</td>
      <td>317</td>
      <td>1301</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>66</td>
      <td>493</td>
      <td>456</td>
      <td>11</td>
      <td>332</td>
      <td>16</td>
      <td>7</td>
      <td>1</td>
      <td>3</td>
      <td>2798</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250</td>
      <td>19</td>
      <td>1921</td>
      <td>1696</td>
      <td>104</td>
      <td>2</td>
      <td>19</td>
      <td>2861</td>
      <td>257</td>
      <td>18</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>1463</td>
      <td>130</td>
      <td>2714</td>
      <td>325</td>
      <td>1</td>
      <td>188</td>
      <td>1595</td>
      <td>38</td>
      <td>105</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>282</td>
      <td>356</td>
      <td>4531</td>
      <td>459</td>
      <td>833</td>
      <td>618</td>
      <td>1</td>
      <td>2627</td>
      <td>253</td>
      <td>87</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>43</td>
      <td>208</td>
      <td>351</td>
      <td>741</td>
      <td>85</td>
      <td>1904</td>
      <td>28</td>
      <td>721</td>
      <td>590</td>
      <td>490</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>66</td>
      <td>23</td>
      <td>122</td>
      <td>213</td>
      <td>519</td>
      <td>1</td>
      <td>1846</td>
      <td>507</td>
      <td>1365</td>
      <td>328</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>126</td>
      <td>102</td>
      <td>2</td>
      <td>306</td>
      <td>102</td>
      <td>14</td>
      <td>2940</td>
      <td>550</td>
      <td>14</td>
      <td>520</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>66</td>
      <td>355</td>
      <td>5</td>
      <td>2633</td>
      <td>1503</td>
      <td>121</td>
      <td>10</td>
      <td>77</td>
      <td>12</td>
      <td>208</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>127</td>
      <td>2</td>
      <td>16</td>
      <td>17</td>
      <td>34</td>
      <td>4</td>
      <td>26</td>
      <td>19</td>
      <td>1175</td>
      <td>97</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>141</td>
      <td>1</td>
      <td>300</td>
      <td>3</td>
      <td>23</td>
      <td>15</td>
      <td>61</td>
      <td>790</td>
      <td>14</td>
      <td>69</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>68</td>
      <td>132</td>
      <td>1</td>
      <td>443</td>
      <td>2</td>
      <td>784</td>
      <td>1579</td>
      <td>158</td>
      <td>1339</td>
      <td>40</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>140</td>
      <td>220</td>
      <td>4715</td>
      <td>768</td>
      <td>143</td>
      <td>365</td>
      <td>6</td>
      <td>96</td>
      <td>1</td>
      <td>464</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>70</td>
      <td>47</td>
      <td>1</td>
      <td>199</td>
      <td>44</td>
      <td>254</td>
      <td>4</td>
      <td>499</td>
      <td>429</td>
      <td>220</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>151</td>
      <td>507</td>
      <td>263</td>
      <td>675</td>
      <td>1</td>
      <td>1</td>
      <td>1388</td>
      <td>1662</td>
      <td>1387</td>
      <td>4418</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>87</td>
      <td>761</td>
      <td>3192</td>
      <td>1661</td>
      <td>1014</td>
      <td>398</td>
      <td>44</td>
      <td>2569</td>
      <td>580</td>
      <td>4215</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>86</td>
      <td>2</td>
      <td>287</td>
      <td>1025</td>
      <td>44</td>
      <td>763</td>
      <td>64</td>
      <td>202</td>
      <td>236</td>
      <td>4657</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>68</td>
      <td>386</td>
      <td>32</td>
      <td>1</td>
      <td>1614</td>
      <td>87</td>
      <td>12</td>
      <td>149</td>
      <td>7</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>115</td>
      <td>595</td>
      <td>44</td>
      <td>536</td>
      <td>132</td>
      <td>390</td>
      <td>298</td>
      <td>2010</td>
      <td>921</td>
      <td>10</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>164</td>
      <td>355</td>
      <td>2</td>
      <td>365</td>
      <td>1</td>
      <td>44</td>
      <td>845</td>
      <td>15</td>
      <td>195</td>
      <td>654</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>21</th>
      <td>67</td>
      <td>184</td>
      <td>3</td>
      <td>4</td>
      <td>26</td>
      <td>328</td>
      <td>1392</td>
      <td>1212</td>
      <td>13</td>
      <td>117</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>68</td>
      <td>16</td>
      <td>7</td>
      <td>342</td>
      <td>1046</td>
      <td>351</td>
      <td>1078</td>
      <td>1</td>
      <td>37</td>
      <td>7</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>70</td>
      <td>2</td>
      <td>279</td>
      <td>464</td>
      <td>361</td>
      <td>3053</td>
      <td>12</td>
      <td>2</td>
      <td>2081</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>135</td>
      <td>5</td>
      <td>1220</td>
      <td>102</td>
      <td>192</td>
      <td>14</td>
      <td>1743</td>
      <td>2076</td>
      <td>3941</td>
      <td>1533</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>43</td>
      <td>936</td>
      <td>519</td>
      <td>1110</td>
      <td>4498</td>
      <td>876</td>
      <td>329</td>
      <td>1664</td>
      <td>229</td>
      <td>403</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>26</th>
      <td>51</td>
      <td>171</td>
      <td>154</td>
      <td>249</td>
      <td>1294</td>
      <td>1650</td>
      <td>67</td>
      <td>57</td>
      <td>2800</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>28</td>
      <td>531</td>
      <td>174</td>
      <td>30</td>
      <td>1</td>
      <td>160</td>
      <td>289</td>
      <td>122</td>
      <td>229</td>
      <td>444</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>77</td>
      <td>325</td>
      <td>4452</td>
      <td>139</td>
      <td>51</td>
      <td>482</td>
      <td>1</td>
      <td>215</td>
      <td>65</td>
      <td>198</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>81</td>
      <td>51</td>
      <td>12</td>
      <td>2806</td>
      <td>961</td>
      <td>244</td>
      <td>49</td>
      <td>11</td>
      <td>2</td>
      <td>61</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24970</th>
      <td>78</td>
      <td>4081</td>
      <td>1580</td>
      <td>3736</td>
      <td>77</td>
      <td>261</td>
      <td>776</td>
      <td>101</td>
      <td>3</td>
      <td>45</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24971</th>
      <td>64</td>
      <td>818</td>
      <td>1987</td>
      <td>2036</td>
      <td>1</td>
      <td>1245</td>
      <td>176</td>
      <td>234</td>
      <td>1033</td>
      <td>13</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24972</th>
      <td>143</td>
      <td>1</td>
      <td>28</td>
      <td>226</td>
      <td>3</td>
      <td>1</td>
      <td>706</td>
      <td>1609</td>
      <td>1</td>
      <td>1465</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24973</th>
      <td>121</td>
      <td>589</td>
      <td>1</td>
      <td>2</td>
      <td>333</td>
      <td>71</td>
      <td>50</td>
      <td>1173</td>
      <td>248</td>
      <td>335</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24974</th>
      <td>78</td>
      <td>29</td>
      <td>2</td>
      <td>138</td>
      <td>135</td>
      <td>1810</td>
      <td>3892</td>
      <td>1</td>
      <td>7</td>
      <td>3653</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24975</th>
      <td>73</td>
      <td>17</td>
      <td>28</td>
      <td>35</td>
      <td>135</td>
      <td>19</td>
      <td>198</td>
      <td>689</td>
      <td>53</td>
      <td>766</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24976</th>
      <td>141</td>
      <td>184</td>
      <td>299</td>
      <td>287</td>
      <td>365</td>
      <td>1</td>
      <td>58</td>
      <td>3278</td>
      <td>559</td>
      <td>315</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24977</th>
      <td>157</td>
      <td>1228</td>
      <td>4430</td>
      <td>34</td>
      <td>1</td>
      <td>213</td>
      <td>849</td>
      <td>213</td>
      <td>15</td>
      <td>1595</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24978</th>
      <td>102</td>
      <td>4323</td>
      <td>1</td>
      <td>3676</td>
      <td>2897</td>
      <td>39</td>
      <td>650</td>
      <td>198</td>
      <td>298</td>
      <td>399</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24979</th>
      <td>65</td>
      <td>100</td>
      <td>1</td>
      <td>120</td>
      <td>36</td>
      <td>1</td>
      <td>1624</td>
      <td>329</td>
      <td>21</td>
      <td>264</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24980</th>
      <td>63</td>
      <td>151</td>
      <td>174</td>
      <td>1</td>
      <td>674</td>
      <td>56</td>
      <td>514</td>
      <td>2843</td>
      <td>844</td>
      <td>2482</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24981</th>
      <td>61</td>
      <td>29</td>
      <td>2</td>
      <td>135</td>
      <td>2</td>
      <td>168</td>
      <td>138</td>
      <td>759</td>
      <td>538</td>
      <td>161</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24982</th>
      <td>209</td>
      <td>434</td>
      <td>171</td>
      <td>2568</td>
      <td>329</td>
      <td>797</td>
      <td>617</td>
      <td>3669</td>
      <td>3681</td>
      <td>1562</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24983</th>
      <td>81</td>
      <td>1</td>
      <td>3356</td>
      <td>176</td>
      <td>179</td>
      <td>878</td>
      <td>1</td>
      <td>582</td>
      <td>105</td>
      <td>119</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24984</th>
      <td>102</td>
      <td>51</td>
      <td>99</td>
      <td>38</td>
      <td>1</td>
      <td>668</td>
      <td>4797</td>
      <td>2149</td>
      <td>187</td>
      <td>888</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24985</th>
      <td>221</td>
      <td>1</td>
      <td>2024</td>
      <td>1</td>
      <td>1</td>
      <td>33</td>
      <td>1</td>
      <td>60</td>
      <td>1089</td>
      <td>1704</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24986</th>
      <td>173</td>
      <td>77</td>
      <td>2</td>
      <td>56</td>
      <td>158</td>
      <td>138</td>
      <td>59</td>
      <td>27</td>
      <td>5</td>
      <td>1205</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24987</th>
      <td>143</td>
      <td>55</td>
      <td>817</td>
      <td>912</td>
      <td>391</td>
      <td>313</td>
      <td>2</td>
      <td>106</td>
      <td>142</td>
      <td>51</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24988</th>
      <td>79</td>
      <td>47</td>
      <td>3</td>
      <td>29</td>
      <td>669</td>
      <td>1862</td>
      <td>361</td>
      <td>1</td>
      <td>158</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24989</th>
      <td>117</td>
      <td>486</td>
      <td>347</td>
      <td>112</td>
      <td>2502</td>
      <td>417</td>
      <td>342</td>
      <td>76</td>
      <td>2734</td>
      <td>489</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24990</th>
      <td>424</td>
      <td>1555</td>
      <td>4737</td>
      <td>87</td>
      <td>4925</td>
      <td>3719</td>
      <td>4429</td>
      <td>1168</td>
      <td>1619</td>
      <td>2711</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24991</th>
      <td>94</td>
      <td>2</td>
      <td>477</td>
      <td>340</td>
      <td>418</td>
      <td>1011</td>
      <td>311</td>
      <td>1419</td>
      <td>535</td>
      <td>1546</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24992</th>
      <td>82</td>
      <td>980</td>
      <td>2</td>
      <td>850</td>
      <td>846</td>
      <td>991</td>
      <td>500</td>
      <td>68</td>
      <td>95</td>
      <td>106</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24993</th>
      <td>62</td>
      <td>779</td>
      <td>573</td>
      <td>2</td>
      <td>304</td>
      <td>521</td>
      <td>164</td>
      <td>307</td>
      <td>773</td>
      <td>1409</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24994</th>
      <td>49</td>
      <td>17</td>
      <td>37</td>
      <td>21</td>
      <td>153</td>
      <td>142</td>
      <td>5</td>
      <td>3</td>
      <td>261</td>
      <td>1117</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24995</th>
      <td>85</td>
      <td>1103</td>
      <td>16</td>
      <td>7</td>
      <td>6</td>
      <td>12</td>
      <td>55</td>
      <td>411</td>
      <td>623</td>
      <td>722</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24996</th>
      <td>62</td>
      <td>28</td>
      <td>135</td>
      <td>710</td>
      <td>5</td>
      <td>393</td>
      <td>25</td>
      <td>143</td>
      <td>238</td>
      <td>318</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24997</th>
      <td>117</td>
      <td>426</td>
      <td>36</td>
      <td>133</td>
      <td>3</td>
      <td>560</td>
      <td>1885</td>
      <td>1</td>
      <td>79</td>
      <td>3681</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24998</th>
      <td>142</td>
      <td>16</td>
      <td>16</td>
      <td>16</td>
      <td>77</td>
      <td>2</td>
      <td>285</td>
      <td>3392</td>
      <td>612</td>
      <td>4382</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24999</th>
      <td>77</td>
      <td>391</td>
      <td>566</td>
      <td>1</td>
      <td>4</td>
      <td>174</td>
      <td>83</td>
      <td>6</td>
      <td>111</td>
      <td>198</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>25000 rows × 501 columns</p>
</div>




```python
# We split the data into chunks and send each chunk seperately, accumulating the results.

def predict(data, rows=512):
    split_array = np.array_split(data, int(data.shape[0] / float(rows) + 1))
    predictions = np.array([])
    for array in split_array:
        predictions = np.append(predictions, predictor.predict(array))
    
    return predictions
```


```python
predictions = predict(test_X.values)
predictions = [round(num) for num in predictions]
```


```python
from sklearn.metrics import accuracy_score
accuracy_score(test_y, predictions)
```




    0.85572



**Question:** How does this model compare to the XGBoost model you created earlier? Why might these two models perform differently on this dataset? Which do *you* think is better for sentiment analysis?

**Answer:**

I believe XGboost is a bit better just because is faster to train and provide a bit better score, NN are great to but I will use them for other kind of problems here they work fine, but the dataset was not that big, XGBOOST will do it fine too

### (TODO) More testing

We now have a trained model which has been deployed and which we can send processed reviews to and which returns the predicted sentiment. However, ultimately we would like to be able to send our model an unprocessed review. That is, we would like to send the review itself as a string. For example, suppose we wish to send the following review to our model.


```python
test_review = 'The simplest pleasures in life are the best, and this film is one of them. Combining a rather basic storyline of love and adventure this movie transcends the usual weekend fair with wit and unmitigated charm.'
```

The question we now need to answer is, how do we send this review to our model?

Recall in the first section of this notebook we did a bunch of data processing to the IMDb dataset. In particular, we did two specific things to the provided reviews.
 - Removed any html tags and stemmed the input
 - Encoded the review as a sequence of integers using `word_dict`
 
In order process the review we will need to repeat these two steps.

**TODO**: Using the `review_to_words` and `convert_and_pad` methods from section one, convert `test_review` into a numpy array `test_data` suitable to send to our model. Remember that our model expects input of the form `review_length, review[500]`.


```python
# TODO: Convert test_review into a form usable by the model and save the results in test_data
test_review_words= review_to_words(test_review)
data_test, datatest_len = convert_and_pad_data(word_dict, test_review_words)
test_data = pd.concat([pd.DataFrame(datatest_len), pd.DataFrame(data_test)], axis=1)
test_data
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>490</th>
      <th>491</th>
      <th>492</th>
      <th>493</th>
      <th>494</th>
      <th>495</th>
      <th>496</th>
      <th>497</th>
      <th>498</th>
      <th>499</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1440</td>
      <td>1219</td>
      <td>704</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>1440</td>
      <td>1219</td>
      <td>704</td>
      <td>1</td>
      <td>1</td>
      <td>1100</td>
      <td>1209</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4</td>
      <td>1219</td>
      <td>1</td>
      <td>1044</td>
      <td>704</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>425</td>
      <td>704</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>1044</td>
      <td>1</td>
      <td>1219</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>3</td>
      <td>1</td>
      <td>1497</td>
      <td>704</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>894</td>
      <td>1</td>
      <td>1</td>
      <td>425</td>
      <td>1</td>
      <td>1497</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>6</td>
      <td>1209</td>
      <td>1</td>
      <td>1</td>
      <td>1768</td>
      <td>704</td>
      <td>1209</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>5</td>
      <td>425</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>894</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1209</td>
      <td>1</td>
      <td>1219</td>
      <td>1</td>
      <td>1497</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>4</td>
      <td>1219</td>
      <td>1</td>
      <td>1729</td>
      <td>704</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1729</td>
      <td>704</td>
      <td>1497</td>
      <td>1</td>
      <td>1100</td>
      <td>1209</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>1729</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>9</td>
      <td>1</td>
      <td>1209</td>
      <td>1</td>
      <td>1497</td>
      <td>1</td>
      <td>894</td>
      <td>704</td>
      <td>1497</td>
      <td>1</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>5</td>
      <td>1100</td>
      <td>1</td>
      <td>1100</td>
      <td>1</td>
      <td>1219</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>7</td>
      <td>1805</td>
      <td>704</td>
      <td>704</td>
      <td>1919</td>
      <td>704</td>
      <td>1497</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>4</td>
      <td>1044</td>
      <td>1</td>
      <td>1</td>
      <td>1209</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>3</td>
      <td>1805</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>7</td>
      <td>1100</td>
      <td>1497</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1120</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5</td>
      <td>894</td>
      <td>1768</td>
      <td>1</td>
      <td>1209</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>20 rows × 501 columns</p>
</div>



Now that we have processed the review, we can send the resulting array to our model to predict the sentiment of the review.


```python
predictor.predict(test_data)
```




    array([0.41449305, 0.6095664 , 0.22298452, 0.60055226, 0.37881207,
           0.46699932, 0.65931064, 0.6342982 , 0.4447666 , 0.61909926,
           0.2806013 , 0.77662194, 0.36736047, 0.65775687, 0.6104962 ,
           0.6002356 , 0.4772657 , 0.46058404, 0.6733759 , 0.61782634],
          dtype=float32)



Since the return value of our model is close to `1`, we can be certain that the review we submitted is positive.

### Delete the endpoint

Of course, just like in the XGBoost notebook, once we've deployed an endpoint it continues to run until we tell it to shut down. Since we are done using our endpoint for now, we can delete it.


```python
estimator.delete_endpoint()
```

## Step 6 (again) - Deploy the model for the web app

Now that we know that our model is working, it's time to create some custom inference code so that we can send the model a review which has not been processed and have it determine the sentiment of the review.

As we saw above, by default the estimator which we created, when deployed, will use the entry script and directory which we provided when creating the model. However, since we now wish to accept a string as input and our model expects a processed review, we need to write some custom inference code.

We will store the code that we write in the `serve` directory. Provided in this directory is the `model.py` file that we used to construct our model, a `utils.py` file which contains the `review_to_words` and `convert_and_pad` pre-processing functions which we used during the initial data processing, and `predict.py`, the file which will contain our custom inference code. Note also that `requirements.txt` is present which will tell SageMaker what Python libraries are required by our custom inference code.

When deploying a PyTorch model in SageMaker, you are expected to provide four functions which the SageMaker inference container will use.
 - `model_fn`: This function is the same function that we used in the training script and it tells SageMaker how to load our model.
 - `input_fn`: This function receives the raw serialized input that has been sent to the model's endpoint and its job is to de-serialize and make the input available for the inference code.
 - `output_fn`: This function takes the output of the inference code and its job is to serialize this output and return it to the caller of the model's endpoint.
 - `predict_fn`: The heart of the inference script, this is where the actual prediction is done and is the function which you will need to complete.

For the simple website that we are constructing during this project, the `input_fn` and `output_fn` methods are relatively straightforward. We only require being able to accept a string as input and we expect to return a single value as output. You might imagine though that in a more complex application the input or output may be image data or some other binary data which would require some effort to serialize.

### (TODO) Writing inference code

Before writing our custom inference code, we will begin by taking a look at the code which has been provided.


```python
!pygmentize serve/predict.py
```

    [34mimport[39;49;00m [04m[36margparse[39;49;00m
    [34mimport[39;49;00m [04m[36mjson[39;49;00m
    [34mimport[39;49;00m [04m[36mos[39;49;00m
    [34mimport[39;49;00m [04m[36mpickle[39;49;00m
    [34mimport[39;49;00m [04m[36msys[39;49;00m
    [34mimport[39;49;00m [04m[36msagemaker_containers[39;49;00m
    [34mimport[39;49;00m [04m[36mpandas[39;49;00m [34mas[39;49;00m [04m[36mpd[39;49;00m
    [34mimport[39;49;00m [04m[36mnumpy[39;49;00m [34mas[39;49;00m [04m[36mnp[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch.nn[39;49;00m [34mas[39;49;00m [04m[36mnn[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch.optim[39;49;00m [34mas[39;49;00m [04m[36moptim[39;49;00m
    [34mimport[39;49;00m [04m[36mtorch.utils.data[39;49;00m
    
    [34mfrom[39;49;00m [04m[36mmodel[39;49;00m [34mimport[39;49;00m LSTMClassifier
    
    [34mfrom[39;49;00m [04m[36mutils[39;49;00m [34mimport[39;49;00m review_to_words, convert_and_pad
    
    [34mdef[39;49;00m [32mmodel_fn[39;49;00m(model_dir):
        [33m"""Load the PyTorch model from the `model_dir` directory."""[39;49;00m
        [34mprint[39;49;00m([33m"[39;49;00m[33mLoading model.[39;49;00m[33m"[39;49;00m)
    
        [37m# First, load the parameters used to create the model.[39;49;00m
        model_info = {}
        model_info_path = os.path.join(model_dir, [33m'[39;49;00m[33mmodel_info.pth[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(model_info_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model_info = torch.load(f)
    
        [34mprint[39;49;00m([33m"[39;49;00m[33mmodel_info: {}[39;49;00m[33m"[39;49;00m.format(model_info))
    
        [37m# Determine the device and construct the model.[39;49;00m
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        model = LSTMClassifier(model_info[[33m'[39;49;00m[33membedding_dim[39;49;00m[33m'[39;49;00m], model_info[[33m'[39;49;00m[33mhidden_dim[39;49;00m[33m'[39;49;00m], model_info[[33m'[39;49;00m[33mvocab_size[39;49;00m[33m'[39;49;00m])
    
        [37m# Load the store model parameters.[39;49;00m
        model_path = os.path.join(model_dir, [33m'[39;49;00m[33mmodel.pth[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(model_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model.load_state_dict(torch.load(f))
    
        [37m# Load the saved word_dict.[39;49;00m
        word_dict_path = os.path.join(model_dir, [33m'[39;49;00m[33mword_dict.pkl[39;49;00m[33m'[39;49;00m)
        [34mwith[39;49;00m [36mopen[39;49;00m(word_dict_path, [33m'[39;49;00m[33mrb[39;49;00m[33m'[39;49;00m) [34mas[39;49;00m f:
            model.word_dict = pickle.load(f)
    
        model.to(device).eval()
    
        [34mprint[39;49;00m([33m"[39;49;00m[33mDone loading model.[39;49;00m[33m"[39;49;00m)
        [34mreturn[39;49;00m model
    
    [34mdef[39;49;00m [32minput_fn[39;49;00m(serialized_input_data, content_type):
        [34mprint[39;49;00m([33m'[39;49;00m[33mDeserializing the input data.[39;49;00m[33m'[39;49;00m)
        [34mif[39;49;00m content_type == [33m'[39;49;00m[33mtext/plain[39;49;00m[33m'[39;49;00m:
            data = serialized_input_data.decode([33m'[39;49;00m[33mutf-8[39;49;00m[33m'[39;49;00m)
            [34mreturn[39;49;00m data
        [34mraise[39;49;00m [36mException[39;49;00m([33m'[39;49;00m[33mRequested unsupported ContentType in content_type: [39;49;00m[33m'[39;49;00m + content_type)
    
    [34mdef[39;49;00m [32moutput_fn[39;49;00m(prediction_output, accept):
        [34mprint[39;49;00m([33m'[39;49;00m[33mSerializing the generated output.[39;49;00m[33m'[39;49;00m)
        [34mreturn[39;49;00m [36mstr[39;49;00m(prediction_output)
    
    [34mdef[39;49;00m [32mpredict_fn[39;49;00m(input_data, model):
        [34mprint[39;49;00m([33m'[39;49;00m[33mInferring sentiment of input data.[39;49;00m[33m'[39;49;00m)
    
        device = torch.device([33m"[39;49;00m[33mcuda[39;49;00m[33m"[39;49;00m [34mif[39;49;00m torch.cuda.is_available() [34melse[39;49;00m [33m"[39;49;00m[33mcpu[39;49;00m[33m"[39;49;00m)
        
        [34mif[39;49;00m model.word_dict [35mis[39;49;00m [36mNone[39;49;00m:
            [34mraise[39;49;00m [36mException[39;49;00m([33m'[39;49;00m[33mModel has not been loaded properly, no word_dict.[39;49;00m[33m'[39;49;00m)
        
        [37m# TODO: Process input_data so that it is ready to be sent to our model.[39;49;00m
        [37m#       You should produce two variables:[39;49;00m
        [37m#         data_X   - A sequence of length 500 which represents the converted review[39;49;00m
        [37m#         data_len - The length of the review[39;49;00m
        input_data_to_words = review_to_words(input_data)
        data_X, data_len = convert_and_pad(model.word_dict,input_data_to_words)
    
        [37m# Using data_X and data_len we construct an appropriate input tensor. Remember[39;49;00m
        [37m# that our model expects input data of the form 'len, review[500]'.[39;49;00m
        data_pack = np.hstack((data_len, data_X))
        data_pack = data_pack.reshape([34m1[39;49;00m, -[34m1[39;49;00m)
        
        data = torch.from_numpy(data_pack)
        data = data.to(device)
    
        [37m# Make sure to put the model into evaluation mode[39;49;00m
        model.eval()
        [34mwith[39;49;00m torch.no_grad():
            output = model.forward(data)
    
    
        [37m# TODO: Compute the result of applying the model to the input data. The variable `result` should[39;49;00m
        [37m#       be a numpy array which contains a single integer which is either 1 or 0[39;49;00m
    
        result = np.round(output.numpy())
    
        [34mreturn[39;49;00m result


As mentioned earlier, the `model_fn` method is the same as the one provided in the training code and the `input_fn` and `output_fn` methods are very simple and your task will be to complete the `predict_fn` method. Make sure that you save the completed file as `predict.py` in the `serve` directory.

**TODO**: Complete the `predict_fn()` method in the `serve/predict.py` file.

### Deploying the model

Now that the custom inference code has been written, we will create and deploy our model. To begin with, we need to construct a new PyTorchModel object which points to the model artifacts created during training and also points to the inference code that we wish to use. Then we can call the deploy method to launch the deployment container.

**NOTE**: The default behaviour for a deployed PyTorch model is to assume that any input passed to the predictor is a `numpy` array. In our case we want to send a string so we need to construct a simple wrapper around the `RealTimePredictor` class to accomodate simple strings. In a more complicated situation you may want to provide a serialization object, for example if you wanted to sent image data.


```python
from sagemaker.predictor import RealTimePredictor
from sagemaker.pytorch import PyTorchModel

class StringPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session):
        super(StringPredictor, self).__init__(endpoint_name, sagemaker_session, content_type='text/plain')

model = PyTorchModel(model_data=estimator.model_data,
                     role = role,
                     framework_version='0.4.0',
                     entry_point='predict.py',
                     source_dir='serve',
                     predictor_cls=StringPredictor)
predictor = model.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

    -------------!

### Testing the model

Now that we have deployed our model with the custom inference code, we should test to see if everything is working. Here we test our model by loading the first `250` positive and negative reviews and send them to the endpoint, then collect the results. The reason for only sending some of the data is that the amount of time it takes for our model to process the input and then perform inference is quite long and so testing the entire data set would be prohibitive.


```python
import glob

def test_reviews(data_dir='../data/aclImdb', stop=250):
    
    results = []
    ground = []
    
    # We make sure to test both positive and negative reviews    
    for sentiment in ['pos', 'neg']:
        
        path = os.path.join(data_dir, 'test', sentiment, '*.txt')
        files = glob.glob(path)
        
        files_read = 0
        
        print('Starting ', sentiment, ' files')
        
        # Iterate through the files and send them to the predictor
        for f in files:
            with open(f) as review:
                # First, we store the ground truth (was the review positive or negative)
                if sentiment == 'pos':
                    ground.append(1)
                else:
                    ground.append(0)
                # Read in the review and convert to 'utf-8' for transmission via HTTP
                review_input = review.read().encode('utf-8')
                # Send the review to the predictor and store the results
                results.append(int(float(predictor.predict(review_input))))
                
            # Sending reviews to our endpoint one at a time takes a while so we
            # only send a small number of reviews
            files_read += 1
            if files_read == stop:
                break
            
    return ground, results
```


```python
ground, results = test_reviews()
```

    Starting  pos  files
    Starting  neg  files



```python
from sklearn.metrics import accuracy_score
accuracy_score(ground, results)
```




    0.862



As an additional test, we can try sending the `test_review` that we looked at earlier.


```python
predictor.predict(test_review)
```




    b'1.0'



Now that we know our endpoint is working as expected, we can set up the web page that will interact with it. If you don't have time to finish the project now, make sure to skip down to the end of this notebook and shut down your endpoint. You can deploy it again when you come back.

## Step 7 (again): Use the model for the web app

> **TODO:** This entire section and the next contain tasks for you to complete, mostly using the AWS console.

So far we have been accessing our model endpoint by constructing a predictor object which uses the endpoint and then just using the predictor object to perform inference. What if we wanted to create a web app which accessed our model? The way things are set up currently makes that not possible since in order to access a SageMaker endpoint the app would first have to authenticate with AWS using an IAM role which included access to SageMaker endpoints. However, there is an easier way! We just need to use some additional AWS services.

<img src="Web App Diagram.svg">

The diagram above gives an overview of how the various services will work together. On the far right is the model which we trained above and which is deployed using SageMaker. On the far left is our web app that collects a user's movie review, sends it off and expects a positive or negative sentiment in return.

In the middle is where some of the magic happens. We will construct a Lambda function, which you can think of as a straightforward Python function that can be executed whenever a specified event occurs. We will give this function permission to send and recieve data from a SageMaker endpoint.

Lastly, the method we will use to execute the Lambda function is a new endpoint that we will create using API Gateway. This endpoint will be a url that listens for data to be sent to it. Once it gets some data it will pass that data on to the Lambda function and then return whatever the Lambda function returns. Essentially it will act as an interface that lets our web app communicate with the Lambda function.

### Setting up a Lambda function

The first thing we are going to do is set up a Lambda function. This Lambda function will be executed whenever our public API has data sent to it. When it is executed it will receive the data, perform any sort of processing that is required, send the data (the review) to the SageMaker endpoint we've created and then return the result.

#### Part A: Create an IAM Role for the Lambda function

Since we want the Lambda function to call a SageMaker endpoint, we need to make sure that it has permission to do so. To do this, we will construct a role that we can later give the Lambda function.

Using the AWS Console, navigate to the **IAM** page and click on **Roles**. Then, click on **Create role**. Make sure that the **AWS service** is the type of trusted entity selected and choose **Lambda** as the service that will use this role, then click **Next: Permissions**.

In the search box type `sagemaker` and select the check box next to the **AmazonSageMakerFullAccess** policy. Then, click on **Next: Review**.

Lastly, give this role a name. Make sure you use a name that you will remember later on, for example `LambdaSageMakerRole`. Then, click on **Create role**.

#### Part B: Create a Lambda function

Now it is time to actually create the Lambda function.

Using the AWS Console, navigate to the AWS Lambda page and click on **Create a function**. When you get to the next page, make sure that **Author from scratch** is selected. Now, name your Lambda function, using a name that you will remember later on, for example `sentiment_analysis_func`. Make sure that the **Python 3.6** runtime is selected and then choose the role that you created in the previous part. Then, click on **Create Function**.

On the next page you will see some information about the Lambda function you've just created. If you scroll down you should see an editor in which you can write the code that will be executed when your Lambda function is triggered. In our example, we will use the code below. 

```python
# We need to use the low-level library to interact with SageMaker since the SageMaker API
# is not available natively through Lambda.
import boto3

def lambda_handler(event, context):

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '**ENDPOINT NAME HERE**',    # The name of the endpoint we created
                                       ContentType = 'text/plain',                 # The data format that is expected
                                       Body = event['body'])                       # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : result
    }
```

Once you have copy and pasted the code above into the Lambda code editor, replace the `**ENDPOINT NAME HERE**` portion with the name of the endpoint that we deployed earlier. You can determine the name of the endpoint using the code cell below.


```python
predictor.endpoint
```




    'sagemaker-pytorch-2020-04-22-00-11-45-604'



Once you have added the endpoint name to the Lambda function, click on **Save**. Your Lambda function is now up and running. Next we need to create a way for our web app to execute the Lambda function.

### Setting up API Gateway

Now that our Lambda function is set up, it is time to create a new API using API Gateway that will trigger the Lambda function we have just created.

Using AWS Console, navigate to **Amazon API Gateway** and then click on **Get started**.

On the next page, make sure that **New API** is selected and give the new api a name, for example, `sentiment_analysis_api`. Then, click on **Create API**.

Now we have created an API, however it doesn't currently do anything. What we want it to do is to trigger the Lambda function that we created earlier.

Select the **Actions** dropdown menu and click **Create Method**. A new blank method will be created, select its dropdown menu and select **POST**, then click on the check mark beside it.

For the integration point, make sure that **Lambda Function** is selected and click on the **Use Lambda Proxy integration**. This option makes sure that the data that is sent to the API is then sent directly to the Lambda function with no processing. It also means that the return value must be a proper response object as it will also not be processed by API Gateway.

Type the name of the Lambda function you created earlier into the **Lambda Function** text entry box and then click on **Save**. Click on **OK** in the pop-up box that then appears, giving permission to API Gateway to invoke the Lambda function you created.

The last step in creating the API Gateway is to select the **Actions** dropdown and click on **Deploy API**. You will need to create a new Deployment stage and name it anything you like, for example `prod`.

You have now successfully set up a public API to access your SageMaker model. Make sure to copy or write down the URL provided to invoke your newly created public API as this will be needed in the next step. This URL can be found at the top of the page, highlighted in blue next to the text **Invoke URL**.

## Step 4: Deploying our web app

Now that we have a publicly available API, we can start using it in a web app. For our purposes, we have provided a simple static html file which can make use of the public api you created earlier.

In the `website` folder there should be a file called `index.html`. Download the file to your computer and open that file up in a text editor of your choice. There should be a line which contains **\*\*REPLACE WITH PUBLIC API URL\*\***. Replace this string with the url that you wrote down in the last step and then save the file.

Now, if you open `index.html` on your local computer, your browser will behave as a local web server and you can use the provided site to interact with your SageMaker model.

If you'd like to go further, you can host this html file anywhere you'd like, for example using github or hosting a static site on Amazon's S3. Once you have done this you can share the link with anyone you'd like and have them play with it too!

> **Important Note** In order for the web app to communicate with the SageMaker endpoint, the endpoint has to actually be deployed and running. This means that you are paying for it. Make sure that the endpoint is running when you want to use the web app but that you shut it down when you don't need it, otherwise you will end up with a surprisingly large AWS bill.

**TODO:** Make sure that you include the edited `index.html` file in your project submission.

Now that your web app is working, trying playing around with it and see how well it works.

**Question**: Give an example of a review that you entered into your web app. What was the predicted sentiment of your example review?

**Answer:**

## Reviews from IMDB for the movie Parasite:


I decided to create my IMDB account, just to express this. This movie is completely Overrated; I can't even beleive that this was consider among the best movies of the year... but winning the Oscar? You got to be kidding me... how can you even compare this garbage, to movies like: 1917 or Joker? and besides that, I read so many 10 ratings, this just seems very weird for me... the script is stupid, the acting sometimes ridiculous, the movie tries to deliver a 'hide' messages for our society in a very dumb way, at the end there was no message and no movie... And just to make it clear, I'm not a US citizen, but just trying to create hype because it was a good foreigner production is ridiculous. I can think of a thousand foreigner movies better than this, Completely disapointed by this "academy".

Your review was NEGATIVE!

What an amazing film! Clearly an essay on the class divide. Writing lovable but nasty people is tough, but perfectly executed here.

They quite clearly cross the line, and when I mean "they" I mean bother the working class and the upper class. You could easily argue that "parasite" refers to one family trying to suck another family dry. Or that the upper class is draining the labour of the lower class, and expecting them to be grateful about it too! There is also something about the American Indian theme, a metaphor for dying cultural traditions which are being replaced by modernity? Nature is dog eat dog? Or is it a homage to an "idyllic" past where Native American culture was relatively classless?

Either way a must watch if you can handle subtitles!

Your review was POSITIVE!

### Delete the endpoint

Remember to always shut down your endpoint if you are no longer using it. You are charged for the length of time that the endpoint is running so if you forget and leave it on you could end up with an unexpectedly large bill.


```python
predictor.delete_endpoint()
```
