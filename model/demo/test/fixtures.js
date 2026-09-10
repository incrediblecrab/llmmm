import { FEATURE_NAMES } from "../ranker.js";

export function policy() {
  return {
    schema_version: 1, feature_version: 1, model_type: "recipe-ranking-mlp-browser",
    activation: "tanh", feature_names: [...FEATURE_NAMES], hidden_dim: 1,
    time_features_enabled: true,
    tensors: {
      feature_mask: new Array(20).fill(1),
      "network.0.bias": [0],
      "network.0.weight": [new Array(20).fill(0).map((_, i) => Number(i === 12))],
      "network.2.bias": [0],
      "network.2.weight": [[1]],
    },
  };
}
