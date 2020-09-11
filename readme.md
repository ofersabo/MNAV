# Preprocessing
###
Create a conda env which support the required packeages
use out environment.yml file
create -f environment.yml

### augment the dataset with special tokens before and after the target entities
 
 ##### FewRel dataset
 python create_dataset_with_marked_entities.py val_wiki.json
 python create_dataset_with_marked_entities.py train_wiki.json
 
 ##### Few-Shot TACRED
 python create_dataset_with_marked_entities.py data_few_shot/_test_data.json
 python create_dataset_with_marked_entities.py data_few_shot/new_downsampled_train_data.json
 python create_dataset_with_marked_entities.py data_few_shot/_dev_data.json

These commands generate new datasets in which special tokens are surrounding the target entities.

### generate episodes based on the augmented datasets





