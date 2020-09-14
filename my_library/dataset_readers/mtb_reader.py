from typing import Dict
from typing import List
import json
import logging
import os
import random
from preprocessing_prepeare_sentence import preprocessing
from preprocessing_prepeare_sentence import head_start_token,head_end_token,tail_start_token,tail_end_token
import copy
from overrides import overrides
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.fields import ListField, IndexField, MetadataField, Field

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@DatasetReader.register("NOTA_reader")
class MTBDatasetReader(DatasetReader):
    def __init__(self,
                 bert_model: str,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 debug = None) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # self._tokenizer.add_special_tokens({'additional_special_tokens':[head_start_token,head_end_token,tail_start_token,tail_end_token])
        self.spacy_splitter = SpacyWordSplitter(keep_spacy_tokens=True)
        self.TRAIN_DATA = "meta_train"
        self.TEST_DATA = "meta_test"
        self.tokens_with_markers = "tokens_with_markers"
        self.head_bert = "head_after_bert"
        self.tail_bert = "tail_after_bert"
        self.bert_indexer = preprocessing(bert_model)
        self.do_once = True
        self.debug = debug

        # write host name
        import socket
        hostname = socket.gethostname()
        logger.info("hostname is: %s", hostname)

    @overrides
    def _read(self, file_path):
        with open(cached_path(file_path), "r") as data_file:
            logger.info("Reading instances from json files at: %s", data_file)
            data = json.load(data_file)
            labels = data[1]
            aux = data[2]
            data = data[0]
            ii = 0
            for x, l, a in zip(data, labels,aux):
                yield self.text_to_instance(x, l, a)
                ii += 1
                if self.debug and ii >= self.debug:
                    break

    @overrides
    def text_to_instance(self, data: dict, relation_type: int = None, aux : list = None) -> Instance:  # type: ignore
        N_relations = []
        location_list = []
        all_tokens_sentences = []

        for i, K_examples in enumerate(data[self.TRAIN_DATA]):
            toknized_sentences = []
            sentences_location = []
            clean_text_for_debug = []
            for single_relation in K_examples:
                if len(single_relation) == 2:
                    single_relation = single_relation["normal_sentence"]
                list_of_string = " ".join(single_relation[self.tokens_with_markers])
                tokenized_tokens = self._tokenizer.tokenize(list_of_string)
                field_of_tokens = TextField(tokenized_tokens, self._token_indexers)

                clean_text_for_debug.append(MetadataField(list_of_string))

                head_after_bert_location, tail_after_bert_location = single_relation[self.head_bert], single_relation[self.tail_bert]
                assert type(head_after_bert_location) is int
                assert type(tail_after_bert_location) is int
                locations_of_entities = MetadataField([head_after_bert_location, tail_after_bert_location])

                sentences_location.append(locations_of_entities)
                toknized_sentences.append(field_of_tokens)
                if self.do_once: # debug process
                    check = self.bert_indexer.preprocessing_flow(copy.deepcopy(single_relation))
                    sentence_m, head_location_after_bert, tail_location_after_bert, head_end, tail_end = check
                    assert sentence_m == single_relation[self.tokens_with_markers]
                    assert head_location_after_bert == head_after_bert_location
                    assert tail_location_after_bert == tail_after_bert_location
                    self.do_once = False

            assert len(sentences_location) == len(toknized_sentences) == len(clean_text_for_debug)

            sentences_location, clean_text_for_debug, toknized_sentences = ListField(sentences_location), ListField(clean_text_for_debug), ListField(toknized_sentences)
            all_tokens_sentences.append(clean_text_for_debug),location_list.append(sentences_location), N_relations.append(toknized_sentences)

        assert len(N_relations) == len(location_list) == len(all_tokens_sentences)
        N_relations, location_list, all_tokens_sentences = ListField(N_relations), ListField(location_list),ListField(all_tokens_sentences)
        fields = {'sentences': N_relations, "locations": location_list, "clean_tokens": all_tokens_sentences}
        # fields = {'sentences': N_relations, "locations": location_list}

        '''
        query_part
        '''
        queries_data = data[self.TEST_DATA]
        queries = []
        location_queries = []
        labels = []
        for query_index,query in enumerate(queries_data):
            if relation_type:
                target = relation_type[query_index]
            else:
                target = -1

            if len(query) == 2:
                query = query["normal_sentence"]
            labels.append(IndexField(target,N_relations))
            list_of_string = " ".join(query[self.tokens_with_markers])
            tokenized_tokens = self._tokenizer.tokenize(list_of_string)
            field_of_tokens = TextField(tokenized_tokens, self._token_indexers)
            assert len(query['tokens']) + 4 == len(query['tokens_with_markers'])

            head_after_bert_location, tail_after_bert_location = query[self.head_bert], query[self.tail_bert]
            locations_of_entities = MetadataField([head_after_bert_location, tail_after_bert_location])
            queries.append(field_of_tokens)
            location_queries.append(locations_of_entities)

        queries = ListField(queries)
        location_queries = ListField(location_queries)

        fields['test'] = queries
        fields['test_location'] = location_queries

        assert len(labels) == len(queries_data)
        fields['label'] = ListField(labels)

        if aux:
            user_target_relations, target_relation_names = self.get_aux_data(aux,relation_type)
            fields['user_relations'] = user_target_relations
            fields['gold'] = target_relation_names

        return Instance(fields)

    def get_aux_data(self, aux,targets):
        user_target_relations = aux[0]
        target_relation_names = [user_target_relations[t] if t < len(user_target_relations) else "NOTA" for t in targets]
        target_relation_names = MetadataField(target_relation_names)
        user_target_relations = MetadataField(user_target_relations)
        return user_target_relations,target_relation_names