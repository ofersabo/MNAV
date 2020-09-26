from my_library.models.base_model import *
import numpy as np
from my_library.my_loss_metric import SpecialLoss, NotaNotInsideBest2,F1AllClasses,measure_navs

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register('baseline_many_navs')
class manyNavs(NotaAverage):
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
                 number_of_NAVs: int = 20,
                 path_to_vector: str = "NAV_250_vecs.npy",

                 ) -> None:
        super().__init__(vocab = vocab,text_field_embedder=text_field_embedder,number_of_linear_layers=number_of_linear_layers
                         ,metrics=metrics,skip_connection=skip_connection,regularizer=regularizer,bert_model=bert_model,
                         hidden_dim=hidden_dim,add_distance_from_mean=add_distance_from_mean,drop_out_rate=drop_out_rate,
                         dot_product=dot_product,negative_cosine=negative_cosine,add_loss_nota2queries=add_loss_nota2queries,
                         raise_softmax=raise_softmax)
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
            "measure_navs": measure_navs(number_of_NAVs)
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
        if path_to_vector is None:
            logger.info("random vectors for NAV initialization")
            x = torch.randn([number_of_NAVs,self.bert_type_model['hidden_size']*2],device=self.device,requires_grad=False)
            self.no_relation_vector = Parameter(x, requires_grad=True)
        else:
            logger.info("loaded NAV vectors for file %s",path_to_vector)
            x = np.load(path_to_vector)
            x = x[np.random.choice(x.shape[0], number_of_NAVs, replace=False), :]
            x = torch.from_numpy(x).to(device=self.device, dtype=torch.float)
            x = x[:number_of_NAVs]
            self.no_relation_vector = Parameter(x, requires_grad=True)

    def add_random_vector_to_each_batch(self,values,no_relation):
        x = no_relation.expand([values.size(0), 1, values.size(2)])
        catted = torch.cat((values, x),dim=1)
        return catted

    @overrides
    def forward(self, sentences, locations, test, test_location, clean_tokens=None, test_clean_text=None,
                label=None, user_relations=None, gold=None) -> Dict[str, torch.Tensor]:

        bert_context_for_relation = self.embbedings(sentences)
        batch_size = bert_context_for_relation.size(0)
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

        N_class_scores = self.distance_layer(final_matrix_represnetation, query_after_mlp)
        scores_of_NOTA = self.distance_layer(no_relation_after_mlp.unsqueeze(0), query_after_mlp)
        maximal_NOTA_value,which_vector_maximal = scores_of_NOTA.max(dim=-1,keepdim=True)

        N_class_scores = N_class_scores.view(N_class_scores.size(0) * N_class_scores.size(1), N_class_scores.size(2))
        scores_of_NOTA = maximal_NOTA_value.view(maximal_NOTA_value.size(0) * maximal_NOTA_value.size(1), maximal_NOTA_value.size(2))

        scores = self.modify_scores(N_class_scores, scores_of_NOTA)
        flat_labels = label.view(label.size(0) * label.size(1), 1).squeeze(1)
        output_dict = {}
        if flat_labels[0].item() != -1:
            loss = self.crossEntropyLoss(scores, flat_labels)
            output_dict["loss"] = loss
            if self.add_loss_nota2queries and self.no_relation_vector is not None:
                difference_nota += difference_nota2queries
            self.add_more_loss_terms(difference_nota, output_dict, None)

            self.metrics['acc'](scores, flat_labels)
            self.measure_additional_metric(difference_nota, None)
        scores = scores.view(bert_context_for_relation.size(0), number_queries, N_way + 1)
        if gold and user_relations:
            self.metrics['f1_no_NOTA'](scores,label,[[r for r in episode] for episode in user_relations],self.training)

        self.metrics['measure_navs'](which_vector_maximal)

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

    def add_fixed_value_to_each_score_task(self,values,no_relation):
        catted = torch.cat((values, no_relation),dim=1)
        return catted

    @overrides
    def modify_scores(self,scores,compactness_score):
        return self.add_fixed_value_to_each_score_task(scores, compactness_score)

