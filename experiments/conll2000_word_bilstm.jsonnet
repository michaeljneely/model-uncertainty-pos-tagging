local batch_size = 64;
local embedding_dim = 400;
local encoder_hidden_size = 400;

{
    "dataset_reader": {
        "type": "conll2000"
    },
    "train_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/train.txt"]),
    "test_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/test.txt"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/dev.txt"]),
    "evaluate_on_test": true,
    "model": {
        "type": "simple_tagger",
        "text_field_embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": embedding_dim,
                    "trainable": true
                }
            }
        },
        "encoder": {
            "type": "compose",
            "encoders": [
                {
                    "type": "lstm",
                    "bidirectional": true,
                    "input_size": embedding_dim,
                    "hidden_size": encoder_hidden_size,
                    "num_layers": 3,
                    "dropout": 0.33
                },
                {
                    "type": "bi-feedforward",
                    "feedforward": {
                        "input_dim": 2*encoder_hidden_size,
                        "num_layers": 1,
                        "hidden_dims": encoder_hidden_size,
                        "activations": "elu"
                    }
                }
            ]
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "trainer": {
        "num_epochs": 40,
        "patience": 5,
        "validation_metric": "+accuracy",
        "optimizer": {
            "type": "adam",
            "weight_decay": 6e-6,
            "lr": 0.002
        }
    }
}
