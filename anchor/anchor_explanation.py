from __future__ import print_function
import io


class AnchorExplanation:
    """Object returned by explainers"""
    def __init__(self, type_, exp_map, as_html):
        self.type = type_
        self.exp_map = exp_map
        self.as_html_fn = as_html

    def names(self, partial_index=None):
        """
        Returns a list of the names of the anchor conditions.
        Args:
            partial_index (int): lets you get the anchor until a certain index.
            For example, if the anchor is (A=1,B=2,C=2) and partial_index=1,
            this will return ["A=1", "B=2"]
        """
        names = self.exp_map['names']
        if partial_index is not None:
            names = names[:partial_index + 1]
        return names

    def features(self, partial_index=None):
        """
        Returns a list of the features used in the anchor conditions.
        Args:
            partial_index (int): lets you get the anchor until a certain index.
            For example, if the anchor uses features (1, 2, 3) and
            partial_index=1, this will return [1, 2]
        """
        features = self.exp_map['feature']
        if partial_index is not None:
            features = features[:partial_index + 1]
        return features

    def precision(self, partial_index=None):
        """
        Returns the anchor precision (a float)
        Args:
            partial_index (int): lets you get the anchor precision until a
            certain index. For example, if the anchor has precisions
            [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5
        """
        precision = self.exp_map['precision']
        if len(precision) == 0:
            return self.exp_map['all_precision']
        if partial_index is not None:
            return precision[partial_index]
        else:
            return precision[-1]

    def coverage(self, partial_index=None):
        """
        Returns the anchor coverage (a float)
        Args:
            partial_index (int): lets you get the anchor coverage until a
            certain index. For example, if the anchor has coverages
            [0.1, 0.5, 0.95] and partial_index=1, this will return 0.5
        """
        coverage = self.exp_map['coverage']
        if len(coverage) == 0:
            return 1
        if partial_index is not None:
            return coverage[partial_index]
        else:
            return coverage[-1]

    def examples(self, only_different_prediction=False,
                 only_same_prediction=False, partial_index=None):
        """
        Returns examples covered by the anchor.
        Args:
            only_different_prediction(bool): if true, will only return examples
            where the anchor  makes a different prediction than the original
            model
            only_same_prediction(bool): if true, will only return examples
            where the anchor makes the same prediction than the original
            model
            partial_index (int): lets you get the examples from the partial
            anchor until a certain index.
        """
        if only_different_prediction and only_same_prediction:
            print('Error: you can\'t have only_different_prediction \
and only_same_prediction at the same time')
            return []
        key = 'covered'
        if only_different_prediction:
            key = 'covered_false'
        if only_same_prediction:
            key = 'covered_true'
        size = len(self.exp_map['examples'])
        idx = partial_index if partial_index is not None else size - 1
        if idx < 0 or idx > size:
            return []
        return self.exp_map['examples'][idx][key]

    def as_html(self, **kwargs):
        return self.as_html_fn(self.exp_map, **kwargs)

    def show_in_notebook(self, **kwargs):
        from IPython.core.display import display, HTML
        out = self.as_html(**kwargs)
        display(HTML(out))

    def save_to_file(self, file_path, **kwargs):
        out = self.as_html(**kwargs)
        io.open(file_path, 'w').write(out)
