from inference.noise import GaussianNoise, FeatureSelectiveNoise
from inference.precision import TruncateDecimals, CastToInt, Quantize
from inference.structure import ShuffleFeatures, ScaleFeatures, RemoveFeatures, FeatureSwap
from inference.corruption import ZeroOut, InsertNaN
from inference.outliers import InjectOutliers
from inference.distraction import AddDummyFeatures, DuplicateFeatures


#Opção 2 – Adicionar validação semântica na SmartParameterPerturber
#  Sobre input externo (gramática de regras)

#     Sim, é totalmente válido e útil ter um .json ou .yaml com exceções, estratégias ou ranges predefinidos.

# Mas isso pode vir na v1.1, para manter a V1 enxuta e funcional.
#  Extensão futura (não na v1.0, mas prevista)

#     Parser de YAML/JSON externo com regras

#     Análise de impacto da mutação

#     Estratégias por perfil de algoritmo (ex: KNN é sensível a int)

#  Alternativas mais inteligentes (futuro):

#     Adicionar regras semânticas ao SmartParameterPerturber (ex: range mínimo para max_depth)

#     Implementar validações pós-inferência baseadas no model_class

#     Criar um dicionário de restrições manuais por parâmetro

# Deseja que eu atualize o SmartParameterPerturber para suportar ranges válidos por tipo/nome de parâmetro?

# remover ignore_rules={"tol"}) e tratar os valores para sempre respeitarem as regras de range

#Implementar níveis de severidade da inferência (leve, média, pesada)

#Expandir o SmartParameterPerturber com regras baseadas em ranges válidos

class InferenceEngine:
    def __init__(self, config):
        self.pipeline = []

        if config.get('noise_level'):
            self.pipeline.append(GaussianNoise(config['noise_level']))
        if config.get('feature_selective_noise'):
            level, features = config['feature_selective_noise']
            self.pipeline.append(FeatureSelectiveNoise(level, features))
        if config.get('truncate_decimals') is not None:
            self.pipeline.append(TruncateDecimals(config['truncate_decimals']))
        if config.get('cast_to_int'):
            self.pipeline.append(CastToInt())
        if config.get('quantize_bins'):
            self.pipeline.append(Quantize(config['quantize_bins']))
        if config.get('shuffle_fraction'):
            self.pipeline.append(ShuffleFeatures(config['shuffle_fraction']))
        if config.get('scale_range'):
            self.pipeline.append(ScaleFeatures(config['scale_range']))
        if config.get('zero_out_fraction'):
            self.pipeline.append(ZeroOut(config['zero_out_fraction']))
        if config.get('insert_nan_fraction'):
            self.pipeline.append(InsertNaN(config['insert_nan_fraction']))
        if config.get('outlier_fraction'):
            self.pipeline.append(InjectOutliers(config['outlier_fraction']))
        if config.get('add_dummy_features'):
            self.pipeline.append(AddDummyFeatures(config['add_dummy_features']))
        if config.get('duplicate_features'):
            self.pipeline.append(DuplicateFeatures(config['duplicate_features']))
        if config.get('remove_features'):
            self.pipeline.append(RemoveFeatures(config['remove_features']))
        if config.get('feature_swap'):
            self.pipeline.append(FeatureSwap(config['feature_swap']))

    def apply(self, X_train, X_test):
        for transform in self.pipeline:
            X_train = transform.apply(X_train)
            X_test = transform.apply(X_test)
        return X_train, X_test
