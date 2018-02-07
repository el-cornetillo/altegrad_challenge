from collections import defaultdict
import numpy as np
import pandas as pd

DUP_THRESHOLD = 0.7
NOT_DUP_THRESHOLD = 0.4
MAX_UPDATE = 0.2
DUP_UPPER_BOUND = 0.9999999
NOT_DUP_LOWER_BOUND = 0.0000001

df_train = pd.read_csv("data/train.csv")
df_test = pd.read_csv("data/test.csv")

def post_process(test_label, path, REPEAT = 2)
    print("Updating the predictions of the pairs with common duplicates..")
    for i in range(REPEAT):
        dup_neighbors = defaultdict(set)

        for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
            if dup:
                dup_neighbors[q1].add(q2)
                dup_neighbors[q2].add(q1)

        for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
            if dup > DUP_THRESHOLD:
                dup_neighbors[q1].add(q2)
                dup_neighbors[q2].add(q1)

        count = 0
        for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
            dup_neighbor_count = len(dup_neighbors[q1].intersection(dup_neighbors[q2]))
            if dup_neighbor_count > 0 and test_label[index] < DUP_UPPER_BOUND:
                update = min(MAX_UPDATE, (DUP_UPPER_BOUND - test_label[index]) / 2)
                test_label[index] += update
                count += 1

        print("Updated:", count)

    print("Updating the predictions of the pairs with common non-duplicates..")
    for i in range(REPEAT):
        not_dup_neighbors = defaultdict(set)

        for dup, q1, q2 in zip(df_train["is_duplicate"], df_train["question1"], df_train["question2"]):
            if not dup:
                not_dup_neighbors[q1].add(q2)
                not_dup_neighbors[q2].add(q1)

        for dup, q1, q2 in zip(test_label, df_test["question1"], df_test["question2"]):
            if dup < NOT_DUP_THRESHOLD:
                not_dup_neighbors[q1].add(q2)
                not_dup_neighbors[q2].add(q1)

        count = 0
        for index, (q1, q2) in enumerate(zip(df_test["question1"], df_test["question2"])):
            dup_neighbor_count = len(not_dup_neighbors[q1].intersection(not_dup_neighbors[q2]))
            if dup_neighbor_count > 0 and test_label[index] > NOT_DUP_LOWER_BOUND:
                update = min(MAX_UPDATE, (test_label[index] - NOT_DUP_LOWER_BOUND) / 2)
                test_label[index] -= update
                count += 1

        print("Updated:", count)

    #submission = pd.DataFrame({"test_id":df_test["test_id"], "is_duplicate":test_label})
    #submission.to_csv("predictions/submission.csv", index=False)

    submission = pd.DataFrame({"Id":df_test["test_id"], "Score":test_label})
    submission.to_csv(path, index=False)


