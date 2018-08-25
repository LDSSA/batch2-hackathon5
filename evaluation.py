import pandas as pd
import ml_metrics as metrics


def read_file(filename, sep):
    with open(filename) as f:
        content = f.readlines()

    songs = {}
    for l in content:
        l = l.strip().split(sep)
        songs[l[0]] = [int(i) for i in l[1:]]

    return songs


def evaluate():
    test_songs = read_file('y_true.txt', ' ')
    pred_songs = read_file('results.csv', ',')

    actual = []
    predicted = []

    for user_id in test_songs.keys():
        actual.append(test_songs[user_id])
        predicted.append(pred_songs[user_id])

    return metrics.mapk(actual, predicted, k=500)
