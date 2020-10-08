local cuda = [0,1,2,3];
local bert_type = 'bert-base-cased';
local which_model = "many_navs";
local models = {
"many_navs":"baseline_many_navs",
};

local train_data = {
"50K_1shot": "TACRED_episodes/train_5w_1s_3q_50K_seed_123.json",
"50K_5shot": "TACRED_episodes/train_5w_5s_1q_50K_seed_123.json",

# revisited
"REVISITED_50K_1_shot": "",
"REVISITED_50K_5_shot": "",


};
local dev_data = {
"50K_1shot": "TACRED_episodes/dev_5w_1s_3q_10K_seed_123.json",
"50K_5shot": "TACRED_episodes/dev_5w_5s_3q_10K_seed_123.json",


# revisited 3
"REVISITED_50K_1_shot": "",
"REVISITED_50K_5_shot": "",


};


local test_data = {
"50K_1shot":"~/NOTA/data/TACRED/test_episodes/split_3_5_way_1_shotall_merged_3q.json",
"50K_5shot":"~/NOTA/data/TACRED/test_episodes/split_3_5_way_5_shot.json",

# revisited 3
"REVISITED_50K_1_shot": "",
"REVISITED_50K_5_shot": "",

};

local setup = "50K_5shot";
local LR = 0.00001;
local bert_type = 'bert-base-cased';
local instances_per_epoch = 700;
local batch_size = 1;
local seed = {"50K_1shot":301191,"50K_5shot":1568};
{
"random_seed":seed[setup],
"numpy_seed":seed[setup],
"pytorch_seed":seed[setup],
  "dataset_reader": {
    "type": "NOTA_reader",
    "bert_model": bert_type,
    "lazy": false,
    "tokenizer": {
      "type": "word",
      "word_splitter": {
        "type": "my-bert-basic-tokenizer",
        "do_lower_case": false
      }
    },
    "token_indexers": {
      "bert": {
          "type": "bert-pretrained",
          "pretrained_model": bert_type,
          "do_lowercase": false,
          "use_starting_offsets": false
      }
    }
  },
  "train_data_path": train_data[setup] ,
  "validation_data_path": dev_data[setup],
  "test_data_path": test_data[setup],
  "evaluate_on_test":true,
  "model": {
    "type": models[which_model],
    "bert_model": bert_type,
    [if which_model == "many_navs" then "path_to_vector"]: "few_shot_tacred_MNAV_250_vecs.npy",
    "hidden_dim": 2000,
    "raise_softmax": -1,
    "dot_product": true,
    "add_distance_from_mean": if which_model == "NAV" then true else false,
    "number_of_linear_layers": 2,
    "drop_out_rate": 0.1,
    "skip_connection": true,
    "negative_cosine":false,
    "text_field_embedder": {
        "allow_unmatched_keys": true,
        "embedder_to_indexer_map": {

            "bert": ["bert"]
        },
        "token_embedders": {
            "bert": {
              "type": "bert-pretrained",
              "pretrained_model":  bert_type,
              "top_layer_only": true,
              "requires_grad": true
            }
        }
    },
    "regularizer": [[".*no_relation.*", {"type": "l2", "alpha": 1e-03}],["liner_layer", {"type": "l2", "alpha": 1e-03}], [".*", {"type": "l2", "alpha": 1e-07}]]

  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size,
    "instances_per_epoch": instances_per_epoch
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": 5,
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": LR
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+m_f1",
    "num_epochs": 20,
    "cuda_device": cuda
  }
}