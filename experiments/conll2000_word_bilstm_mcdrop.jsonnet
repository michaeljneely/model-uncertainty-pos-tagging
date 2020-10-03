local batch_size = 64;
local embedding_dim = 300;
local encoder_hidden_size = 400;

{
    "dataset_reader": {
        "type": "conll2000",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            },
            "token_characters": {
                "type": "characters",
                "character_tokenizer": {
                    "byte_encoding": "utf-8"
                }
            }
        }
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
                    "type": "summed-embedding",
                    "token_embedders":
                    [
                        {
                            "type": "embedding",
                            "embedding_dim": embedding_dim,
                            "trainable": true
                        },
                        {
                            "type": "embedding",
                            "pretrained_file": "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
                            "embedding_dim": embedding_dim,
                            "trainable": false
                        }
                    ]
                },
                "token_characters": "empty"
            }
        },
        "encoder": {
            "type": "compose",
            "encoders": [
                {
                    "type": "variational-dropout",
                    "p": 0.33,
                    "input_dim": embedding_dim
                },
                {
                    "type": "stacked_bidirectinoal_lstm",
                    "input_size": embedding_dim,
                    "hidden_size": encoder_hidden_size,
                    "num_layers": 3,
                    "recurrent_dropout_probability": 0.33,
                    "layer_dropout_probability": 0.33
                },
                {
                    "type": "variational-dropout",
                    "p": 0.33,
                    "input_dim": 2 * encoder_hidden_size 
                },
                {
                    "type": "bi-feedforward",
                    "feedforward": {
                        "input_dim": 2 * encoder_hidden_size,
                        "num_layers": 1,
                        "hidden_dims": encoder_hidden_size,
                        "activations": "elu"
                    }
                }
            ]
        },
        "initializer": {
            "regexes": [
                [
                    "encoder.encoder1._feedforward._linear_layers.0.weight", 
                    {
                        "type": "normal",
                        "mean": 0,
                        "std": 1,
                    }
                ],
                ["text_field_embedder.token_embedder_tokens.embed_0.weight", "zero"]
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
