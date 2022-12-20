import yaml

from word_vectors.dataset_per_document import *
from word_vectors.model import *

debug_flag = False

if __name__ == '__main__':
    pl.seed_everything(0)

    with open('../defaults.yaml', 'r') as file:
        defaults = yaml.safe_load(file)

    datamodule = DataModule(**defaults.get('datamodule'), valence_only=defaults.get('valence_only'))
    train_dataset = datamodule.train_set.dataset
    model = LSTM_Classifier(**defaults.get('model'), n_input_features=train_dataset[0][0].size(-1), output_size=len(train_dataset.label_encoder.categories_[0]) - int(train_dataset.label_encoder.drop_idx_[0] is not None))

    trainer = pl.Trainer(
        **defaults.get('trainer'),
        fast_dev_run=debug_flag,
    )

    trainer.fit(model, datamodule=datamodule)
    ts_model = model.to_torchscript(method="trace", example_inputs=train_dataset[0][0].unsqueeze(0))
    torch.jit.save(ts_model, "../artifacts/model.torchscript")