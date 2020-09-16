# MNAV model

## Preprocessing
First create a conda env which support the required packeages
use out environment.yml file
* command: create -f environment.yml

### Augment the dataset with special tokens before and after the target entities
These commands generate new datasets in which special tokens are surrounding the target entities. 
 ##### FewRel dataset
1. python create_dataset_with_marked_entities.py val_wiki.json
2. python create_dataset_with_marked_entities.py train_wiki.json
 
 ##### Few-Shot TACRED
1. python create_dataset_with_marked_entities.py data_few_shot/_test_data.json
2. python create_dataset_with_marked_entities.py data_few_shot/new_downsampled_train_data.json
3. python create_dataset_with_marked_entities.py data_few_shot/_dev_data.json



## Generate episodes based on the augmented datasets

##### FewRel dataset
1. python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 10 1 50 5 456 FEWREL_data/episodes/train_10w_1s_5q_100K_50_NOTA_rate_seed_456.json &
2. python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 5 5 50 5 456 FEWREL_data/episodes/train_5w_5s_5q_100K_50_NOTA_rate_seed_456.json &
3. python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 1 50 5 456 ./FEWREL_data/episodes/DEV_5w_1s_5q_10K_50_NOTA_rate_seed_456.json
4. python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 5 50 5 456 ./FEWREL_data/episodes/DEV_5w_5s_5q_10K_50_NOTA_rate_seed_456.json


## Train the Few-Shot MNAV model

 ##### FewRel model
* 1 shot model: allennlp train experiments/FEWREL_1_shot.jsonnet -s results/fewrel/1_shot/ --include-package my_library
* 5 shot model: allennlp train experiments/FEWREL_5_shot.jsonnet -s results/fewrel/5_shot/ --include-package my_library
 
 ##### Few-Shot TACRED




