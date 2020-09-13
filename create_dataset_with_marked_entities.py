from preprocessing_prepeare_sentence import preprocessing
import json
import copy
import sys
model_name = 'bert-base-cased'
dataset = sys.argv[1] if len(sys.argv) > 1 else "FEWREL_data/val_wiki.json"
bert_model = sys.argv[2] if len(sys.argv) > 2 else model_name
pre = preprocessing(bert_model)
# for f in ["data/fewrel_val.json","data/fewrel_train.json"]:
for f in [dataset]:
    output_dataset = f[:f.rfind(".")] + "_markers.json"
    print(output_dataset)
    total_relatoin = {}
    data = json.load(open(f))
    for relation_type in data:
        this_realtion_list = []
        for x in data[relation_type]:
            two_option_per_sentence = {}
            sentence_info = pre.preprocessing_flow(copy.deepcopy(x))
            tokens_with_markers, h_start, t_start, h_end, t_end = sentence_info

            assert h_start < h_end
            assert t_start < t_end
            assert type(h_end) is int
            assert type(t_end) is int
            this_instance_dict = copy.deepcopy(x)
            this_instance_dict["head_after_bert"] = h_start
            this_instance_dict["tail_after_bert"] = t_start
            this_instance_dict["tokens_with_markers"] = tokens_with_markers
            this_instance_dict["head_end"] = h_end
            this_instance_dict["tail_end"] = t_end
            this_realtion_list.append(this_instance_dict)
        total_relatoin[relation_type] = this_realtion_list
    json.dump(total_relatoin,open(output_dataset,"w"))