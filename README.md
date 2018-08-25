# batch2-hackathon5

[Description](https://docs.google.com/document/d/1JeuxcWkFMZBHE3xZkM2OVcLKoxNk2bfg7aU9hzsqFvs/edit?usp=sharing)

[Leaderboard]()

## Test evaluations

When creating training / test sets, you'll have a few different ways to evaluate your scores:

### via the command line

If you create a train / test set and write them out to a file you can get a score with the following:

```bash
python evaluation.py ./y_true.txt ./y_pred.txt
```

And it will print out your score.

### Calling our wrapper function

You can supply two python dictionaries to the `evaluate` function:

```py

from evaluation import evaluate

y_true = {
    'ba986b9277b96cd6de07dd07be8362b67b764dd4': [12985, 304842, ...]
}

y_pred = {
    'ba986b9277b96cd6de07dd07be8362b67b764dd4': [12985, 304842, ...]
}

# note that the order is VERY important!
evaluate(y_true, y_pred)
```

### Using the mapk function directly

You can see how we are using the `ml_metrics.mapk(actual, predicted, k=500)` by looking in evaluation.py if you need
further flexibility.

# Making submissions

## tl;dr

make a prediction for all users in `data/test_users.csv` and upload a csv of it

## slightly longer explanation

You'll submit a csv with no header via the normal upload. The first entry must always be the ID of the user and the rest
are song ids.

```bash
ba986b9277b96cd6de07dd07be8362b67b764dd4,12985,304842,...
42a9daa28f605e4f269711946cdbe0498a172706,217471,177172,...
...
```

You are required to produce predictions for every single user in `data/test_users.csv`.

