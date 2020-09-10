from . import utils
from . import anchor_base
from . import anchor_explanation
import numpy as np
import json
import os
import string
import sys
from io import open
import numpy as np

def id_generator(size=15):
    """Helper function to generate random div ids. This is useful for embedding
    HTML into ipython notebooks."""
    chars = list(string.ascii_uppercase + string.digits)
    return ''.join(np.random.choice(chars, size, replace=True))

def exp_normalize(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

class TextGenerator(object):
    def __init__(self, url=None):
        from transformers import DistilBertTokenizer, DistilBertForMaskedLM
        import torch
        self.torch = torch
        self.url = url
        if url is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.bert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            self.bert = DistilBertForMaskedLM.from_pretrained('distilbert-base-cased')
            self.bert.to(self.device)
            self.bert.eval()

    def unmask(self, text_with_mask):
        torch = self.torch
        tokenizer = self.bert_tokenizer
        model = self.bert
        encoded = np.array(tokenizer.encode(text_with_mask, add_special_tokens=True))
        input_ids = torch.tensor(encoded)
        masked = (input_ids == self.bert_tokenizer.mask_token_id).numpy().nonzero()[0]
        to_pred = torch.tensor([encoded], device=self.device)
        with torch.no_grad():
            outputs = model(to_pred)[0]
        ret = []
        for i in masked:
            v, top_preds = torch.topk(outputs[0, i], 500)
            words = tokenizer.convert_ids_to_tokens(top_preds)
            v = np.array([float(x) for x in v])
            ret.append((words, v))
        return ret

class SentencePerturber:
    def __init__(self, words, tg, onepass=False):
        self.tg = tg
        self.words = words
        self.cache = {}
        self.mask = self.tg.bert_tokenizer.mask_token
        self.array = np.array(words, '|U80')
        self.onepass = onepass
        self.pr = np.zeros(len(self.words))
        for i in range(len(words)):
            a = self.array.copy()
            a[i] = self.mask
            s = ' '.join(a)
            w, p = self.probs(s)[0]
            self.pr[i] =  min(0.5, dict(zip(w, p)).get(words[i], 0.01))
    def sample(self, data):
        a = self.array.copy()
        masks = np.where(data == 0)[0]
        a[data != 1] = self.mask
        if self.onepass:
            s = ' '.join(a)
            rs = self.probs(s)
            reps = [np.random.choice(a, p=p) for a, p in rs]
            a[masks] = reps
        else:
            for i in masks:
                s = ' '.join(a)
                words, probs = self.probs(s)[0]
                a[i] = np.random.choice(words, p=probs)
        return a

    def probs(self, s):
        if s not in self.cache:
            r = self.tg.unmask(s)
            self.cache[s] = [(a, exp_normalize(b)) for a, b in r]
            if not self.onepass:
                self.cache[s] = self.cache[s][:1]
        return self.cache[s]


    def perturb_sentence(present, n, prob_change=0.5):
        raw = np.zeros((n, len(self.words)), '|U80')
        data = np.ones((n, len(self.words)))




class AnchorText(object):
    """bla"""
    def __init__(self, nlp, class_names, use_unk_distribution=True, mask_string='UNK'):
        """
        Args:
            nlp: spacy object
            class_names: list of strings
            use_unk_distribution: if True, the perturbation distribution
                will just replace words randomly with mask_string.
                If False, words will be replaced by similar words using word
                embeddings
            mask_string: String used to mask tokens if use_unk_distribution is True.
        """
        self.nlp = nlp
        self.class_names = class_names
        self.use_unk_distribution = use_unk_distribution
        self.tg = None
        self.mask_string = mask_string
        if not self.use_unk_distribution:
            self.tg = TextGenerator()

    def get_sample_fn(self, text, classifier_fn, onepass=False, use_proba=False):
        true_label = classifier_fn([text])[0]
        processed = self.nlp(text)
        words = np.array([x.text for x in processed], dtype='|U80')
        positions = [x.idx for x in processed]
        # positions = list(range(len(words)))
        perturber = None
        if not self.use_unk_distribution:
            perturber = SentencePerturber(words, self.tg, onepass=onepass)
        def sample_fn(present, num_samples, compute_labels=True):
            if self.use_unk_distribution:
                data = np.ones((num_samples, len(words)))
                raw = np.zeros((num_samples, len(words)), '|U80')
                raw[:] = words
                for i, t in enumerate(words):
                    if i in present:
                        continue
                    n_changed = np.random.binomial(num_samples, .5)
                    changed = np.random.choice(num_samples, n_changed,
                                               replace=False)
                    raw[changed, i] = self.mask_string
                    data[changed, i] = 0
                raw_data = [' '.join(x) for x in raw]
            else:
                data = np.zeros((num_samples, len(words)))
                for i in range(len(words)):
                    if i in present:
                        continue
                    probs = [1 - perturber.pr[i], perturber.pr[i]]
                    data[:, i] = np.random.choice([0, 1], num_samples, p=probs)
                data[:, present] = 1
                raw_data = []
                for i, d in enumerate(data):
                    r = perturber.sample(d)
                    data[i] = r == words
                    raw_data.append(' '.join(r))
            labels = []
            if compute_labels:
                labels = (classifier_fn(raw_data) == true_label).astype(int)
            labels = np.array(labels)
            max_len = max([len(x) for x in raw_data])
            dtype = '|U%d' % (max(80, max_len))
            raw_data = np.array(raw_data, dtype).reshape(-1, 1)
            return raw_data, data, labels
        return words, positions, true_label, sample_fn

    def explain_instance(self, text, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=10, onepass=False,
                          use_proba=False, beam_size=4,
                          **kwargs):
        if type(text) == bytes:
            text = text.decode()
        words, positions, true_label, sample_fn = self.get_sample_fn(
            text, classifier_fn, onepass=onepass, use_proba=use_proba)
        # print words, true_label
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample_fn, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, stop_on_first=True,
            coverage_samples=1, **kwargs)
        exp['names'] = [words[x] for x in exp['feature']]
        exp['positions'] = [positions[x] for x in exp['feature']]
        exp['instance'] = text
        exp['prediction'] = true_label
        explanation = anchor_explanation.AnchorExplanation('text', exp,
                                                           self.as_html)
        return explanation

    def as_html(self, exp):
        predict_proba = np.zeros(len(self.class_names))
        exp['prediction'] = int(exp['prediction'])
        predict_proba[exp['prediction']] = 1
        predict_proba = list(predict_proba)

        def jsonize(x):
            return json.dumps(x)
        this_dir, _ = os.path.split(__file__)
        bundle = open(os.path.join(this_dir, 'bundle.js'), encoding='utf8').read()
        random_id = 'top_div' + id_generator()

        example_obj = []

        def process_examples(examples, idx):
            idxs = exp['feature'][:idx + 1]
            out_dict = {}
            new_names = {'covered_true': 'coveredTrue', 'covered_false': 'coveredFalse', 'covered': 'covered'}
            for name, new in new_names.items():
                ex = [x[0] for x in examples[name]]
                out = []
                for e in ex:
                    processed = self.nlp(str(e))
                    raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction']) for i in idxs]
                    out.append({'text': e, 'rawIndexes': raw_indexes})
                out_dict[new] = out
            return out_dict

        example_obj = []
        for i, examples in enumerate(exp['examples']):
            example_obj.append(process_examples(examples, i))

        explanation = {'names': exp['names'],
                       'certainties': exp['precision'] if len(exp['precision']) else [exp['all_precision']],
                       'supports': exp['coverage'],
                       'allPrecision': exp['all_precision'],
                       'examples': example_obj}
        processed = self.nlp(exp['instance'])
        raw_indexes = [(processed[i].text, processed[i].idx, exp['prediction'])
                       for i in exp['feature']]
        raw_data = {'text': exp['instance'], 'rawIndexes': raw_indexes}
        jsonize(raw_indexes)

        out = u'''<html>
        <meta http-equiv="content-type" content="text/html; charset=UTF8">
        <head><script>%s </script></head><body>''' % bundle
        out += u'''
        <div id="{random_id}" />
        <script>
            div = d3.select("#{random_id}");
            lime.RenderExplanationFrame(div,{label_names}, {predict_proba},
            {true_class}, {explanation}, {raw_data}, "text", "anchor");
        </script>'''.format(random_id=random_id,
                            label_names=jsonize(self.class_names),
                            predict_proba=jsonize(list(predict_proba)),
                            true_class=jsonize(False),
                            explanation=jsonize(explanation),
                            raw_data=jsonize(raw_data))
        out += u'</body></html>'
        return out

    def show_in_notebook(self, exp, true_class=False, predict_proba_fn=None):
        """Bla"""
        out = self.as_html(exp, true_class, predict_proba_fn)
        from IPython.core.display import display, HTML
        display(HTML(out))
