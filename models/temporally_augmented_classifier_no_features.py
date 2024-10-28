from models.temporally_augmented_classifier import TemporallyAugmentedClassifier


class TemporallyAugmentedClassifierNoFeatures(TemporallyAugmentedClassifier):
    def __init__(self, **kwargs):
        super(TemporallyAugmentedClassifierNoFeatures, self).__init__(**kwargs)

    def _extend_with_old_labels(self, x, use_predictions=False):
        if not use_predictions:
            old_labels = self._old_labels
        else:
            old_labels = self._old_predictions
        ext = range(0, self.num_old_labels)
        x_ext = {}
        for el, old_label in zip(ext, list(old_labels)):
            x_ext[el] = old_label
        return x_ext
