from my_library.models.no_relation_avarage_NOTA_model import *



@Model.register('convert_to_vector')
class MultipleNOTA(NotaAverage):
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
                 between_clusters_distance: bool = False,
                 drop_out_rate: float = 0.2,
                 dot_product: bool = True

                 ) -> None:
        super().__init__(vocab=vocab,text_field_embedder=text_field_embedder,number_of_linear_layers=number_of_linear_layers,
                         metrics=metrics,skip_connection=skip_connection,regularizer=regularizer,
                         bert_model=bert_model,hidden_dim=hidden_dim,add_distance_from_mean=add_distance_from_mean,
                         between_clusters_distance=between_clusters_distance,drop_out_rate=drop_out_rate
                         ,dot_product=dot_product)


    # def forward(self,  sentences, locations):
    #     bert_context_for_relation = self.embbedings(sentences)
    #     bert_represntation = self.extract_vectors_from_markers(bert_context_for_relation, locations)
    #
    #     after_mlp_aggregated = self.go_thorugh_mlp(bert_represntation,self.first_liner_layer,self.second_liner_layer).to(self.device)
    #     NOTA = self.go_thorugh_mlp(self.no_relation_vector,self.first_liner_layer,self.second_liner_layer).to(self.device)
    #     return {"vector":after_mlp_aggregated,"NOTA":NOTA}