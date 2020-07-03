// Configuration for a RoBERTa sentiment analysis classifier, using the binary
// Stanford Sentiment Treebank (Socher at al. 2013).

local transformer_model = "roberta-base";
local transformer_dim = 768;
local cls_is_last_token = false;
local dataset = "beer";

{
  "dataset_reader": {
    "type": "custom_text_csv",
    "token_indexers": {
      "tokens": {
        "type": "pretrained_transformer",
        "model_name": transformer_model
      }
    }
  },
  "train_data_path": "",
  "validation_data_path": "",
  "test_data_path": "",
  "model": {
    "type": "basic_classifier",
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
          "type": "pretrained_transformer",
          "model_name": transformer_model
        }
      }
    },
    "seq2vec_encoder": {
       "type": "cls_pooler",
       "embedding_dim": transformer_dim,
       "cls_is_last_token": cls_is_last_token
    },
    "dropout": 0.1
  },
  "data_loader": {
    "batch_sampler": {
      "type": "bucket",
      "sorting_keys": ["tokens"],
      "batch_size" : 8
    }
  },
  "trainer": {
    "num_epochs": 1,
    "cuda_device" : 0,
    "validation_metric": "+accuracy",
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": 10,
      "num_steps_per_epoch": 3088,
      "cut_frac": 0.06
    },
    "optimizer": {
      "type": "huggingface_adamw",
      "lr": 2e-5,
      "weight_decay": 0.1,
    }
  }
}
