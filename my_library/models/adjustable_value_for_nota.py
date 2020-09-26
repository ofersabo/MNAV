from my_library.models.base_model import *



@Model.register('nota_scalar')
class NOTAScalar(NotaAverage):
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
                 negative_cosine:bool =False,
                 raise_softmax:int = 2,
                 number_of_NAVs: int = 20,
                 path_to_vector: str = "NAV_250_vecs.npy",
                 ) -> None:
        super().__init__(vocab=vocab,text_field_embedder=text_field_embedder,number_of_linear_layers=number_of_linear_layers,
                         metrics=metrics,skip_connection=skip_connection,regularizer=regularizer,
                         bert_model=bert_model,hidden_dim=hidden_dim,add_distance_from_mean=add_distance_from_mean,
                         drop_out_rate=drop_out_rate
                         ,dot_product=dot_product,negative_cosine=negative_cosine,
                         raise_softmax=raise_softmax)

        self.nota_value = torch.tensor([190.0],device=self.device, requires_grad=True)
        self.nota_value = Parameter(self.nota_value, requires_grad=True)
        self.no_relation_vector = None


    def add_fixed_value_to_each_score_task(self,values,no_relation):
        x = no_relation.expand([values.size(0), 1])
        catted = torch.cat((values, x),dim=1)
        return catted

    @overrides
    def add_random_vector_to_each_batch(self,final_matrix_represnetation, no_relation_after_mlp):
        #ignore this function
        return final_matrix_represnetation

    @overrides
    def modify_scores(self,scores,compactness_score):
        return self.add_fixed_value_to_each_score_task(scores, self.nota_value)

    @overrides
    def distance_to_average(self, vector):
        return torch.tensor([0.0])
