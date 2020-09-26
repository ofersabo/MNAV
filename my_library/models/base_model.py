from typing import Dict, List
import logging
import allennlp
import torch
from overrides import overrides
from allennlp.modules import TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.span_extractors import EndpointSpanExtractor
from allennlp.training.metrics import CategoricalAccuracy,F1Measure
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from torch.nn.parameter import Parameter
from allennlp.nn import util
from torch.nn.functional import softmax,normalize
from my_library.my_loss_metric import SpecialLoss, NotaNotInsideBest2,F1AllClasses

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


BERT_LARGE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                     "hidden_act": "gelu",
                     "hidden_dropout_prob": 0.1,
                     "hidden_size": 1024,
                     "initializer_range": 0.02,
                     "intermediate_size": 4096,
                     "max_position_embeddings": 512,
                     "num_attention_heads": 16,
                     "num_hidden_layers": 24,
                     "type_vocab_size": 2,
                     "vocab_size": 30522
                    }

BERT_BASE_CONFIG = {"attention_probs_dropout_prob": 0.1,
                    "hidden_act": "gelu",
                    "hidden_dropout_prob": 0.1,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": 3072,
                    "max_position_embeddings": 512,
                    "num_attention_heads": 12,
                    "num_hidden_layers": 12,
                    "type_vocab_size": 2,
                    "vocab_size": 30522
                   }
linear = "linear"

@Model.register('base_model')
class NotaAverage(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 number_of_linear_layers : int = 2,
                 metrics: Dict[str, allennlp.training.metrics.Metric] = None,
                 skip_connection: bool = False,
                 regularizer: RegularizerApplicator = None,
                 bert_model: str = None,
                 hidden_dim: int = 500,
                 add_distance_from_mean: bool = True,
                 drop_out_rate: float = 0.2,
                 dot_product: bool = True,
                 negative_cosine:  bool = False,
                 add_loss_nota2queries = False,
                 raise_softmax: int = 1,
                 ) -> None:
        super().__init__(vocab,regularizer)
        self.embbedings = text_field_embedder
        self.bert_type_model = BERT_BASE_CONFIG if "base" in bert_model else BERT_LARGE_CONFIG
        self.extractor = EndpointSpanExtractor(input_dim=self.bert_type_model['hidden_size'], combination="x,y")
        self.crossEntropyLoss   = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.metrics = metrics or {
            # "NOTA_NotInBest2": NotaNotInsideBest2(),
            "acc": CategoricalAccuracy(),
            "loss_of_NOTA": SpecialLoss(),
            "f1_no_NOTA": F1AllClasses(),
        }
        if negative_cosine:
            self.metrics["neg_cosine"] = SpecialLoss()
        self.negative_cos = negative_cosine
        self.dot_product = dot_product
        self.first_liner_layer = torch.nn.Linear(self.bert_type_model['hidden_size']*2,hidden_dim)
        self.second_liner_layer = torch.nn.Linear(hidden_dim,hidden_dim)
        self.do_skip_connection = skip_connection
        self.raise_softmax = raise_softmax
        self.number_of_linear_layers = number_of_linear_layers
        self.tanh = torch.nn.Tanh()
        self.drop_layer = torch.nn.Dropout(p=drop_out_rate)
        self.add_distance_from_mean = add_distance_from_mean
        self.add_loss_nota2queries = add_loss_nota2queries
        self.no_relation_vector = torch.randn([1,self.bert_type_model['hidden_size']*2],device=self.device,requires_grad=False)
        self.no_relation_vector = Parameter(self.no_relation_vector, requires_grad=True)

    def distance_to_average(self, vector):
        return torch.log(torch.sqrt(torch.sum(vector ** 2)))

    def add_random_vector_to_each_batch(self,values,no_relation):
        x = no_relation.expand([values.size(0), 1, values.size(2)])
        catted = torch.cat((values, x),dim=1)
        return catted

    def extract_vectors_from_markers(self, embbeds, location):
        stacked_embed = embbeds.view(-1,embbeds.size(-2), embbeds.size(-1))
        pt_tensor_from_list = torch.FloatTensor(location)
        indeces = util.combine_initial_dims(pt_tensor_from_list).long().to(self.device)
        value = self.extractor(stacked_embed, indeces)
        return value

    def average_tensor_of_sentences(self,values_of_all_sentences):
            return values_of_all_sentences.mean(dim=0,keepdim=True).to(self.device)

    def reduce_K_to_mean(self,matrix):
        prototypes = matrix.mean(dim=2).squeeze(2)
        return prototypes

    @overrides
    def forward(self, sentences, locations, test, test_location, clean_tokens=None, test_clean_text=None,
                label=None, user_relations=None, gold=None) -> Dict[str, torch.Tensor]:

        bert_context_for_relation = self.embbedings(sentences)
        N_way = bert_context_for_relation.size(1)
        K_shot = bert_context_for_relation.size(2)
        bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, locations)

        after_mlp_aggregated = self.go_thorugh_mlp(bert_represntation, self.first_liner_layer,
                                                   self.second_liner_layer).to(self.device)
        difference_nota = self.return_zero_torch()
        if self.no_relation_vector is not None:
            no_relation_after_mlp = self.go_thorugh_mlp(self.no_relation_vector, self.first_liner_layer,
                                                        self.second_liner_layer).to(self.device)
            difference_nota = self.distance_to_average(
                no_relation_after_mlp - self.average_tensor_of_sentences(after_mlp_aggregated))

        after_mlp = after_mlp_aggregated.view(bert_context_for_relation.size(0), N_way, K_shot,
                                              after_mlp_aggregated.size(-1))

        final_matrix_represnetation = self.reduce_K_to_mean(after_mlp)
        if self.no_relation_vector is not None:
            final_matrix_represnetation = self.add_random_vector_to_each_batch(final_matrix_represnetation,
                                                                               no_relation_after_mlp)

        ''' query matrix'''
        query_representation = self.embbedings(test)
        query_matrix = self.extract_vectors_from_markers(query_representation, test_location)
        number_queries = query_matrix.size(0) // query_representation.size(0)
        query_after_mlp_aggregated = self.go_thorugh_mlp(query_matrix, self.first_liner_layer,
                                                         self.second_liner_layer).to(self.device)
        query_after_mlp = query_after_mlp_aggregated.view(query_representation.size(0), number_queries, -1)
        if self.no_relation_vector is not None:
            difference_nota2queries = no_relation_after_mlp - self.average_tensor_of_sentences(
                query_after_mlp_aggregated)
            difference_nota2queries = self.distance_to_average(difference_nota2queries)

        scores = self.distance_layer(final_matrix_represnetation, query_after_mlp)
        compactness_score = self.compute_compactness(K_shot, after_mlp, scores, user_relations)
        negative_cosine = self.compute_negative_cosine(scores, after_mlp, query_after_mlp, N_way, K_shot)
        scores = scores.view(scores.size(0) * scores.size(1), scores.size(2))
        scores = self.modify_scores(scores, compactness_score)
        flat_labels = label.view(label.size(0) * label.size(1), 1).squeeze(1)
        output_dict = {}
        if flat_labels[0].item() != -1:
            loss = self.crossEntropyLoss(scores, flat_labels)
            output_dict["loss"] = loss
            if self.add_loss_nota2queries and self.no_relation_vector is not None:
                difference_nota += difference_nota2queries
            self.add_more_loss_terms(difference_nota, output_dict, negative_cosine)

            self.metrics['acc'](scores, flat_labels)
            self.measure_additional_metric(difference_nota, negative_cosine)
        scores = scores.view(bert_context_for_relation.size(0), number_queries, N_way + 1)
        if gold and user_relations:
            self.metrics['f1_no_NOTA'](scores,label,[[r for r in episode] for episode in user_relations],self.training)

        output_dict["scores"] = scores
        return output_dict

    def measure_additional_metric(self, difference_nota,negative_cosine):
        if self.add_distance_from_mean:
            self.metrics['loss_of_NOTA'](difference_nota)
        if self.negative_cos:
            self.metrics['neg_cosine'](negative_cosine)


    def add_more_loss_terms(self, difference_nota, output_dict,negative_cosine):
        if self.add_distance_from_mean:
            output_dict["loss"] += difference_nota
        if self.negative_cos:
            output_dict["loss"] += negative_cosine

    def distance_layer(self, final_matrix_represnetation, query_after_mlp):
        if self.dot_product:
            test_matrix = query_after_mlp
            tensor_of_matrices = final_matrix_represnetation.permute(0, 2, 1)
            scores = torch.matmul(test_matrix, tensor_of_matrices).to(self.device)
        else:
            test_matrix = query_after_mlp.unsqueeze(1)
            delta = test_matrix - final_matrix_represnetation
            distance = torch.norm(delta, dim=-1)
            scores = -1 * distance

        return scores

    def go_thorugh_mlp(self, concat_represntentions,first_layer,second_layer):
        after_drop_out_layer = self.drop_layer(concat_represntentions)
        after_first_layer = first_layer(after_drop_out_layer)
        x = self.tanh(after_first_layer)
        x = second_layer(x)
        if self.do_skip_connection:
            x = x + after_first_layer

        return x

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metric_results = {}
        for metric_name, metric in self.metrics.items():
            if "f1_no_NOTA" == metric_name:
                f1_measure_metric = metric.get_metric(reset=reset)
                metric_results["m_p"] = f1_measure_metric[0]
                metric_results["m_r"] = f1_measure_metric[1]
                metric_results["m_f1"] = f1_measure_metric[2]
            else:
                metric_results[metric_name] = metric.get_metric(reset)

        return metric_results

    def modify_scores(self,scores,value_to_add):
        return scores

    def return_zero_torch(self):
        return torch.tensor([0.0]).to(self.device)
