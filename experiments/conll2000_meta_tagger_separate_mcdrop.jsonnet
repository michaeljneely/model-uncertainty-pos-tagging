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
                    "byte_encoding": "utf-8",
                    "end_tokens": [32]
                }
            }
        },
        "tag_label": "pos"
    },
    "train_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/train.txt"]),
    "test_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/test.txt"]),
    "validation_data_path": std.join("/", [std.extVar("PWD"), "datasets/conll2000/dev.txt"]),
    "evaluate_on_test": true,
    "model": {
        "type": "meta_tagger_wrapper",
        "component_models": {
            "character": {
                "type": "component_tagger",
                "text_field_embedder": {
                    "token_embedders": {
                        "tokens": "empty", 
                        "token_characters": {
                            "type": "sentence_character_encoding",
                            "embedding": {
                                "embedding_dim": embedding_dim,
                                "trainable": true
                            },
                            "encoder": {
                                "type": "compose",
                                "encoders": [
                                    {
                                        "type": "variational-dropout",
                                        "p": 0.05,
                                        "input_dim": embedding_dim,
                                    },
                                    {
                                        "type": "stacked_bidirectional_lstm",
                                        "input_size": embedding_dim,
                                        "hidden_size": encoder_hidden_size,
                                        "num_layers": 3,
                                        "recurrent_dropout_probability": 0.33,
                                        "layer_dropout_probability": 0.33
                                    }
                                ]   
                            }
                        }
                    }
                },
                "encoder": {
                    "type": "compose",
                    "encoders": [
                        {
                            "type": "variational-dropout",
                            "p": 0.33,
                            "input_dim": 4 * encoder_hidden_size,
                            "bidirectional": false 
                        },
                        {
                            "type": "feedforward",
                            "feedforward": {
                                "input_dim": 4 * encoder_hidden_size,
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
                        [
                            "text_field_embedder.token_embedder_tokens.embed_0.weight",
                            {
                                "type": "normal",
                                "mean": 0,
                                "std": 1,
                            }
                        ]
                    ]
                }
            },
            "word": {
                "type": "component_tagger",
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
                            "type": "stacked_bidirectional_lstm",
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
            }
        },
        "meta_model": {
            "type": "meta_tagger",
            "encoder": {
                "type": "compose",
                "encoders": [
                    {
                        "type": "stacked_bidirectional_lstm",
                        "input_size": 2 * encoder_hidden_size,
                        "hidden_size": encoder_hidden_size,
                        "num_layers": 1,
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
                            "input_dim": encoder_hidden_size * 2,
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
                    ]
                ],
            }
        }
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size
        }
    },
    "trainer": {
        "type": "meta",
        "cuda_device": 0,
        "num_epochs": 40,
        "validation_metric": "+accuracy",
        "patience": 5,
        "moving_average": {
            "type": "exponential",
            "decay": 0.999994
        },
        "component_optimizers": {
            "character": {
                "type": "gradient_descent",
                "optimizer": {
                    "type": "adam",
                    "lr": 2e-3,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "amsgrad": true
                }
            },
            "word": {
                "optimizer": {
                    "type": "adam",
                    "lr": 2e-3,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "amsgrad": true
                }
            },
            "meta": {
                "optimizer": {
                    "type": "adam",
                    "lr": 2e-3,
                    "betas": [0.9, 0.999],
                    "eps": 1e-08,
                    "amsgrad": true
                }
            }
        }
    },
    "uncertainty_experiment": {
        "batch_size": 64,
        "predictor_type": "mc_dropout_sentence_tagger"
    },
}
