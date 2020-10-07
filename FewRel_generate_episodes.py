import json
import random
import numpy as np
import os,sys,inspect
from collections import OrderedDict

filename = sys.argv[1] if len(sys.argv) > 1 else "../fewrel_train_markers.json"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 100000
N = int(sys.argv[3]) if len(sys.argv) > 3 else 10
K = int(sys.argv[4]) if len(sys.argv) > 4 else 1
NOTA_RATE = float(sys.argv[5]) if len(sys.argv) > 5 else float(50)
number_of_queries = int(sys.argv[6]) if len(sys.argv) > 6 else 1
seed = int(sys.argv[7]) if len(sys.argv) > 7 else 123
train_file = sys.argv[8] if len(sys.argv) > 8 else "train_NOTA_1M.json"
print("The seed is ",seed)
random.seed(seed)
np.random.seed(seed)

whole_division = json.load(open(filename),object_pairs_hook=OrderedDict)
relations = list(whole_division.keys())
NOTA_RATE = NOTA_RATE / 100.0


def create_episode(whole_division,relations,N,K,number_of_queries,NOTA_RATE):
    assert "no_relation" not in whole_division
    sampled_relation = random.sample(relations, N)
    meta_train = [random.sample(whole_division[i], K) for i in sampled_relation]

    meta_test_list = []
    target_list = []
    list_target_relations= []
    random_values = np.random.rand(number_of_queries)
    for q_index in range(number_of_queries):
        target = random.choice(range(len(sampled_relation)))
        target_relation = sampled_relation[target]
        if random_values[q_index] < NOTA_RATE:
            # replace
            possible_relation = [r for r in relations if r not in sampled_relation]
            target_relation = random.sample(possible_relation, 1)[0]
            assert not (target_relation in sampled_relation)
            meta_test = random.choice(whole_division[target_relation])
            target = int(N)
        else:
            instance_in_ss = meta_train[target]
            temp = [x for x in whole_division[target_relation] if x not in instance_in_ss]
            meta_test = random.choice(temp)

        list_target_relations.append(target_relation)
        meta_test_list.append(meta_test)
        target_list.append(target)

    return {"meta_train": meta_train, "meta_test": meta_test_list}, target_list,[list_target_relations,sampled_relation]


def relationID_to_name(file_name):
    try:
        with open(file_name, 'r') as fp:
            id2name = json.load(fp)
            name2id = {v: k for k, v in id2name.items()}
    except FileNotFoundError:
        try:
            with open("data/" + file_name, 'r') as fp:
                id2name = json.load(fp)
                name2id = {v: k for k, v in id2name.items()}
        except FileNotFoundError:
            with open("../" + file_name, 'r') as fp:
                id2name = json.load(fp)
                name2id = {v: k for k, v in id2name.items()}
    return id2name, name2id


def main():
    FewRel()


def FewRel():
    episodes = []
    targets_lists = []
    class_relations = []
    for i in range(size):
        # if i % 1000 == 0: print(i)
        episode, targets, [aux_1,aux_0] = create_episode(whole_division, relations, N, K, number_of_queries, NOTA_RATE)
        targets_lists.append(targets)
        episodes.append(episode)
        class_relations.append((aux_0,aux_1))

    final_data = [episodes, targets_lists,class_relations]
    json.dump(final_data, open(train_file, "w"))


if __name__ == "__main__":
    main()

