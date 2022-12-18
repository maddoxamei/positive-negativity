import yaml

from dataset import *
from model import *

debug_flag = True

if __name__ == '__main__':
    with open('../defaults.yaml', 'r') as file:
        defaults = yaml.safe_load(file)

    datamodule = DataModule(**defaults.get('datamodule'))
    train_dataset = datamodule.train_set.dataset
    model = LSTM_Classifier(**defaults.get('model'), n_input_features=train_dataset[0][0].size(-1), output_size=len(train_dataset.label_encoder.categories_[0]) - int(train_dataset.label_encoder.drop_idx_[0] is not None))

    trainer = pl.Trainer(
        **defaults.get('trainer'),
        fast_dev_run=debug_flag,
    )

    trainer.fit(model, datamodule=datamodule)

