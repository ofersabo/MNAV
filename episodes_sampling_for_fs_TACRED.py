import json
import random
import os, sys, inspect, argparse
from numpy.random import choice
import numpy as np
from collections import OrderedDict

NOT_SAME_ENTITIES = "not_same_entities"
important_keys = set(['id', 'relation', 'token', 'tokens', 'h', 't', 'head_after_bert', 'tail_after_bert', 'tokens_with_markers', 'head_end', 'tail_end'])

def remove_relations_with_too_few_instances(data,K):
    relations_to_remove = []
    for r in data:
        if len(data[r]) <= K:
            # we need to remove this relation type as we don't have enough instances.
            print("removed relation: ", r)
            relations_to_remove.append(r)
    data = {k: v for k, v in data.items() if k not in relations_to_remove}
    return data


def normalized_weights(other_list):
    weights = np.array(other_list)
    weights = weights / np.sum(weights)
    weights = weights.tolist()
    return weights


def TACRED_create_episode(all_data, weights_all_relation, uniform_dist_drop_no_relation, N, K,
                          number_of_queries):
    # uniform sampling but remove no_relation
    sampled_relation = choice(a=[*all_data], size=N, replace=False, p=uniform_dist_drop_no_relation).tolist()
    meta_train = [random.sample(all_data[i], K) for i in sampled_relation]

    meta_test_list = []
    target_list = []
    list_relations = []
    targets_relations = choice(a=[*all_data], size=number_of_queries, replace=True, p=weights_all_relation).tolist()
    for t in targets_relations:
        if t in sampled_relation:
            correct_target = sampled_relation.index(t)
            instance_in_ss = meta_train[correct_target]
            if args.possible_same_entities:
                temp = [x for x in all_data[t] if x not in instance_in_ss]
            else:
                possible_instances = [s[NOT_SAME_ENTITIES] for s in instance_in_ss]
                possible_instances = set.intersection(*possible_instances)
                assert len(possible_instances) > 0
                temp = [x for x in all_data[t] if x["id"] in possible_instances]
                assert len(temp) == len(possible_instances)
            single_query = random.choice(temp)
        else:
            correct_target = N
            single_query = random.choice(all_data[t])

        assert type(single_query) is dict
        assert type(correct_target) is int

        meta_test_list.append(single_query)
        target_list.append(correct_target)
        list_relations.append(t)

    # remove unnecessary data
    for class_index,c in enumerate(meta_train):
        for i,this_instance in enumerate(c):
            meta_train[class_index][i] = {k:v for k,v in this_instance.items() if k in important_keys}

    for test_index,t in enumerate(meta_test_list):
        meta_test_list[test_index] = {k:v for k,v in t.items() if k in important_keys}


    return {"meta_train": meta_train, "meta_test": meta_test_list}, target_list, [list_relations,sampled_relation]


def get_weights(all_data):
    weights_all_relation = [len(all_data[r]) for r in all_data]
    weights_all_relation = sum(weights_all_relation)
    return [len(all_data[r]) / weights_all_relation for r in all_data]


def get_query_weights(all_data,w):
    z = w[:]
    no_relation_index = list(all_data.keys()).index("no_relation")
    z[no_relation_index] = 0.0
    z = normalized_weights(z)
    return z


def not_overlapping_entities_set(data):
    for r,instances in data.items():
        if r == "no_relation": continue
        entity_instances = {}
        all_relation_instances = set()
        for relation_insta in instances:
            add_instance_to_entity_set(entity_instances, relation_insta)
            all_relation_instances.add(relation_insta["id"])

        for relation_insta in instances:
            head_entity = relation_insta["h"][0].lower()
            tail_entity = relation_insta["t"][0].lower()
            possible_instances = all_relation_instances - entity_instances[head_entity] - entity_instances[tail_entity]
            assert len(possible_instances) > 1
            relation_insta[NOT_SAME_ENTITIES] = possible_instances


def add_instance_to_entity_set(entity_instances, relation_insta):
    head_entity = relation_insta["h"][0].lower()
    add_entity_to_its_set(entity_instances, head_entity, relation_insta)
    tail_entity = relation_insta["t"][0].lower()
    add_entity_to_its_set(entity_instances, tail_entity, relation_insta)


def add_entity_to_its_set(entity_instances, entity,relation_insta):
    if entity not in entity_instances:
        entity_instances[entity] = set()
    entity_instances[entity].add(relation_insta["id"])


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)

    whole_division = json.load(open(args.file_name))
    assert "no_relation" in whole_division

    whole_division = remove_relations_with_too_few_instances(whole_division,args.K)

    weights_all_relation = get_weights(whole_division)
    query_weights = get_query_weights(whole_division,weights_all_relation)
    uniform_dist_drop_no_relation = [1 / (len(query_weights) - 1) if i > 0.0 else 0.0 for i in query_weights ]

    if not args.possible_same_entities:
        not_overlapping_entities_set(whole_division)

    create_episodes(whole_division, weights_all_relation, uniform_dist_drop_no_relation)


def create_episodes(whole_division, weights_all_relation, uniform_dist_drop_no_relation):
    episodes = []
    targets_lists = []
    aux_data = []
    for i in range(args.episodes_size):
        episode, targets, [targets_relation_names, N_relations] = TACRED_create_episode(whole_division, weights_all_relation,
                                                    uniform_dist_drop_no_relation, args.N, args.K,
                                                    args.number_of_queries)
        targets_lists.append(targets)
        episodes.append(episode)
        aux_data.append((N_relations,targets_relation_names))
    final_data = [episodes, targets_lists,aux_data]
    json.dump(final_data, open(args.output_file_name, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_name", type=str, required=False,default="../Few_Shot_transformation_and_sampling/data_few_shot/_test_data_markers.json")
    parser.add_argument("--episodes_size", default=50000, type=int, required=False,
                        help="The number of episodes to create")
    parser.add_argument("--N", type=int, required=False,default=5,help="How many ways in each episode")
    parser.add_argument("--K", type=int, required=False,default=5,
                        help="How many instances represent each class")
    parser.add_argument("--number_of_queries", default=3, type=int, required=False)
    parser.add_argument("--seed", type=int, required=False,default=160290,
                        help="seed number")
    parser.add_argument("--output_file_name", type=str, required=False,default="my_debug_episdoe",
                        help="The file name to be generated")
    parser.add_argument("--possible_same_entities", action="store_false", required=False)

    global args
    args = parser.parse_args()

    main()
