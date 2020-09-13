# MNAV model
## Preprocessing
Create a conda env which support the required packeages
use out environment.yml file
create -f environment.yml

#### augment the dataset with special tokens before and after the target entities
 
 ##### FewRel dataset
 python create_dataset_with_marked_entities.py val_wiki.json
 python create_dataset_with_marked_entities.py train_wiki.json
 
 ##### Few-Shot TACRED
 python create_dataset_with_marked_entities.py data_few_shot/_test_data.json
 
 python create_dataset_with_marked_entities.py data_few_shot/new_downsampled_train_data.json
 
 python create_dataset_with_marked_entities.py data_few_shot/_dev_data.json

These commands generate new datasets in which special tokens are surrounding the target entities.

## generate episodes based on the augmented datasets

## FewRel dataset
python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 10 1 50 5 456 FEWREL_data/episodes/train_10w_1s_5q_100K_50_NOTA_rate_seed_456.json &

python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 5 5 50 5 456 FEWREL_data/episodes/train_5w_5s_5q_100K_50_NOTA_rate_seed_456.json &


python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 1 50 5 456 ./FEWREL_data/episodes/DEV_5w_1s_5q_10K_50_NOTA_rate_seed_456.json

python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 5 50 5 456 ./FEWREL_data/episodes/DEV_5w_5s_5q_10K_50_NOTA_rate_seed_456.json

