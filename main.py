from data_storage import DataStorage
from feature_generator import FeatureGenerator
import lightgbm

if __name__ == '__main__':
    data_storage = DataStorage(json_dir='update.json')
    generator_feature = FeatureGenerator(data_storage=data_storage)

    features = generator_feature.gen_features(is_train=False)

    model = lightgbm.Booster(model_file='lgbr_base.txt')
    res = model.predict(features)
    print(res)
    


