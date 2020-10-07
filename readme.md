# MNAV model

## 1. Prerequisites 
First create a conda env which support the required packages
use out environment.yml file
```bash
conda env create --name mnav --file=environment.yml 
```
Be aware that the first line of environment.yml defines the env name.

## 2. Preprocessing 

Getting access to the FewRel and Few-Shot TACRED datasets.

FewRel: https://thunlp.github.io/2/fewrel2_da.html

Few-Shot TACRED: https://github.com/ofersabo/Few_Shot_transformation_and_sampling


### Augment the dataset with special tokens before and after the target entities
These commands generate new datasets in which special tokens are surrounding the target entities. 
 ##### FewRel dataset
``` bash 
python create_dataset_with_marked_entities.py val_wiki.json
python create_dataset_with_marked_entities.py train_wiki.json
```
 ##### Few-Shot TACRED
 ```bash 
python create_dataset_with_marked_entities.py data_few_shot/_test_data.json
python create_dataset_with_marked_entities.py data_few_shot/new_downsampled_train_data.json
python create_dataset_with_marked_entities.py data_few_shot/_dev_data.json
```


## 3. Generate episodes based on the augmented datasets

##### FewRel dataset
```bash 
python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 10 1 50 5 456 FEWREL_data/episodes/train_10w_1s_5q_100K_50_NOTA_rate_seed_456.json &
python FewRel_generate_episodes.py.py FEWREL_data/train_wiki_markers.json 100000 5 5 50 5 456 FEWREL_data/episodes/train_5w_5s_5q_100K_50_NOTA_rate_seed_456.json &
python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 1 50 5 456 ./FEWREL_data/episodes/DEV_5w_1s_5q_10K_50_NOTA_rate_seed_456.json
python FewRel_generate_episodes.py.py FEWREL_data/val_wiki_markers.json 10000 5 5 50 5 456 ./FEWREL_data/episodes/DEV_5w_5s_5q_10K_50_NOTA_rate_seed_456.json
```

##### Few-Shot TACRED dataset

To create episodes for Few-shot TACRED, use the script which we published in the Few-Shot TACRED data repository:
https://github.com/ofersabo/Few_Shot_transformation_and_sampling
``` bash 
python episodes_sampling_for_fs_TACRED.py --file_name data_few_shot/_dev_data_markers.json --episodes_size 10000 --N 5 --K 1 --number_of_queries 3 --seed 123 --output_file_name TACRED_episodes/dev_5w_1s_3q_10K_seed_123.json
python episodes_sampling_for_fs_TACRED.py --file_name data_few_shot/_dev_data_markers.json --episodes_size 10000 --N 5 --K 5 --number_of_queries 3 --seed 123 --output_file_name TACRED_episodes/dev_5w_5s_3q_10K_seed_123.json 
python episodes_sampling_for_fs_TACRED.py --file_name data_few_shot/new_downsampled_train_data_markers.json --episodes_size 50000 --N 5 --K 1 --number_of_queries 3 --seed 123 --output_file_name TACRED_episodes/train_5w_1s_3q_50K_seed_123.json 
python episodes_sampling_for_fs_TACRED.py --file_name data_few_shot/new_downsampled_train_data_markers.json --episodes_size 50000 --N 5 --K 5 --number_of_queries 1 --seed 123 --output_file_name TACRED_episodes/train_5w_5s_1q_50K_seed_123.json
```

## 4. Train the Few-Shot MNAV model

 ##### FewRel model
* 1 shot model: 
```bash 
allennlp train experiments/FEWREL_1_shot.jsonnet -s results/fewrel/1_shot/ --include-package my_library
```
* 5 shot model: 
``` bash 
allennlp train experiments/FEWREL_5_shot.jsonnet -s results/fewrel/5_shot/ --include-package my_library
```
 
 ##### Few-Shot TACRED
 * 1 shot model: 
```bash 
allennlp train experiments/Few_Shot_TACRED_1_shot.jsonnet -s results/TACRED/1_shot/MNAV/ --include-package my_library
```
* 5 shot model:
```bash 
allennlp train experiments/Few_Shot_TACRED_5_shot.jsonnet -s results/TACRED/5_shot/MNAV/ --include-package my_library 
```


