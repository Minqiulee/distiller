import distiller
import torch
import torch.nn as nn


def most_similar(weights_column: torch.Tensor,
                 neurons: torch.Tensor) -> Tuple(torch.Tensor, float, float):
    neurons_T = neurons.transpose(0, 1)
    cos = nn.CosineSimilarity(0)
    similarities = [cos(neuron, weights_column).item() for neuron in neurons_T]
    similarities = torch.Tensor(similarities)
    max_similarity, max_neuron_index = torch.max(similarities, 0)
    scale = torch.norm(weights_column) / torch.norm(
        neurons_T[max_neuron_index])

    return neurons_T[max_neuron_index], max_similarity, scale


def get_important_weights(
        layer_name: str, parameter_dict: Dict[str,
                                              torch.Tensor]) -> torch.Tensor:
    return parameter_dict['weight']


def decompose(original_weights: torch.Tensor, important_weights: torch.Tensor,
              threshould: float) -> torch.Tensor:
    """
    [Inputs]
    original_weights: (N[i], N[i+1]) 
    important_weights: (N[i], P[i+1])

    [Outputs]
    scaling_matrix: (P[i+1], N[i+1])
    """

    scaling_matrix = torch.zeros(important_weights.size()[-1],
                                 original_weights.size()[-1])

    for weight in original_weights.transpose(0, -1):
        if weight in important_weights.transpose(0, -1):
            scaling_matrix[important_weights.transpose(0, -1) == weight] = 1
        else:
            most_similar_neuron, similarity, scale = most_similar(
                weight, important_weights)
            most_similar_neuron_index_in_important_weights = important_weights == most_similar_neuron
            if similarity >= threshould:
                scaling_matrix[
                    most_similar_neuron_index_in_important_weights] = scale

    return scaling_matrix


def compensation(module_name: str, original_weights_2: torch.Tensor,
                 scaling_matrix: torch.Tensor) -> torch.Tensor:
    """
    [Inputs]
    original_weights_2: (N[i+2], N[i+1], K, K)
    scaling_matrix: (P[n+1], N[i+1])
    [Outputs]
    new_weights_2: (N[i+2], P[i+1])
    """
    if module_name.startswith('conv'):
        # 2-mode product
        new_weights_2 = torch.tensordot(original_weights_2,
                                        scaling_matrix,
                                        dims=([1], [1]))
    else:
        # Note that when multiplying the next layer, it needs to be transposed
        new_weights_2 = torch.matmul(scaling_matrix, original_weights_2)
    return new_weights_2


def reload_weights(model: nn.Module,
                   compensated_weight_list: List[torch.Tensor]) -> nn.Module:
    for i, layer in enumerate(model.state_dict()):
        model.state_dict()[layer].copy_(compensated_weight_list[i])

    return model


def merge_pruning_compensation(model: nn.Module,
                               threshould: float) -> nn.Module:
    last_name = ""
    last_layer: torch.Tensor = None
    compensated_weight_list = []

    for name, layer in model.named_modules():
        if name:  # otherwise it's the parent node nad we don't need it
            if last_name and last_layer:
                last_parameter_dict = dict(last_layer.named_parameters())

                last_important_weights = get_important_weights(
                    last_name, last_parameter_dict)
                scaling_matrix = decompose(last_parameter_dict['weight'],
                                           last_important_weights, threshould)

                parameter_dict = dict(layer.named_parameters())
                new_weights = compensation(name, parameter_dict['weight'],
                                           scaling_matrix)
            else:
                parameter_dict = dict(layer.named_parameters())
                # Just return this layer's original weight
                new_weights = parameter_dict['weight']

            last_name, last_layer = name, layer
            compensated_weight_list.append(new_weights)

    new_model = reload_weights(model, compensated_weight_list)

    return new_model