local cuda = [0,1,2,3];
//local cuda = [3];
local bert_type = 'bert-base-cased';
local which_model = "many_navs";
local models = {
//"NAV":"average_no_relation",
//"no_average": "average_no_relation",
//"threshold": "nota_scalar" ,
"many_navs":"baseline_many_navs",
};

local train_data = {
"1_shot": "FEWREL_data/episodes/train_10w_1s_5q_100K_50_NOTA_rate_seed_456.json",
"5_shot": "FEWREL_data/episodes/train_5w_5s_5q_100K_50_NOTA_rate_seed_456.json",
};

local dev_data = {
"1_shot": "FEWREL_data/episodes/DEV_5w_1s_5q_10K_50_NOTA_rate_seed_456.json",
"5_shot": "FEWREL_data/episodes/DEV_5w_5s_5q_10K_50_NOTA_rate_seed_456.json",
};
//local bert_type = 'bert-large-cased';
local batch_size = {"1_shot":4,"5_shot":2};
local setup = "1_shot";
local LR = 0.00001;
local instances_per_epoch = 6000;
local seed = {"1_shot":301191,"5_shot":6717};

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
//  "test_data_path":second_dev_data[setup],
  "evaluate_on_test":false,
  "model": {
    "type": models[which_model],
    [if which_model == "many_navs" then "path_to_vector"]: "FEWREL_per_categories_MNAVs.npy",
    "bert_model": bert_type,
    "hidden_dim": 2000,
    "raise_softmax": -1,
    "dot_product": true,
    "add_distance_from_mean": if which_model == "NAV" then true else false,
    "number_of_linear_layers": 2,
    "drop_out_rate": 0.1,
    "skip_connection": true,
    "oracle_for_compactness": false,
    "negative_cosine":false,
//    "add_loss_nota2queries": true,
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
    "batch_size": batch_size[setup],
    "instances_per_epoch": instances_per_epoch
  },
    "validation_iterator": {
    "type": "basic",
    "batch_size": 5,
    "instances_per_epoch": null
  },
  "trainer": {
    "optimizer": {
        "type": "adam",
        "lr": LR
    },
    "num_serialized_models_to_keep": 1,
    "validation_metric": "+m_f1",
    "num_epochs": 50,
//    "grad_norm": 2.0,
    "patience": 100,
    "cuda_device": cuda
  }
}

