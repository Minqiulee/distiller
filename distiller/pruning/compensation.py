import distiller

class CompensatePrunner(object):
    def __init__(self, name, sensitivities, criteria, threshold, bn_lambda, **kwargs):
        self.name = name
        self.sensitivities = sensitivities
        self.criteria = criteria # choose which criterion to use at first masking step
        self.threshold = threshold
        self.bn_lambda = bn_lambda # parameter used after batch normalization layer

    def set_param_mask(self, param, param_name, zeros_mask_dict, meta):
        if param_name not in self.sensitivities:
            if '*' not in self.sensitivities:
                return
            else:
                sensitivity = self.sensitivities['*']
        else:
            sensitivity = self.sensitivities[param_name]

        # first mask the tensor based on sensitivity criterion.
        zeros_mask_dict[param_name].mask = distiller.create_mask_sensitivity_criterion(param, sensitivity)

        # compensate masked elements with elements kept.
        zeros_mask_dict[param_name].mask = distiller.create_mask_similarity_criterion(param, self.threshold, self.bn_lambda)