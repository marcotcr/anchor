from . import anchor_base
import numpy as np
import sklearn


class AnchorImage(object):
    """bla"""
    def __init__(self, distribution_path=None,
                 transform_img_fn=None, n=1000, dummys=None, white=None,
                 segmentation_fn=None):
        """"""
        self.hide = True
        self.white = white
        if segmentation_fn is None:
            from skimage.segmentation import quickshift
            segmentation_fn = lambda x: quickshift(x, kernel_size=4, # noqa
                                                   max_dist=200, ratio=0.2)
        self.segmentation = segmentation_fn
        if dummys is not None:
            self.hide = False
            self.dummys = dummys
        elif distribution_path:
            self.hide = False
            import os
            import skimage

            if not transform_img_fn:
                def transform_img(path):
                    img = skimage.io.imread(path)
                    short_egde = min(img.shape[:2])
                    yy = int((img.shape[0] - short_egde) / 2)
                    xx = int((img.shape[1] - short_egde) / 2)
                    crop_img = img[yy: yy + short_egde, xx: xx + short_egde]
                    return skimage.transform.resize(crop_img, (224, 224))

                def transform_imgs(paths):
                    out = []
                    for i, path in enumerate(paths):
                        if i % 100 == 0:
                            print(i)
                        out.append(transform_img(path))
                    return out
                transform_img_fn = transform_imgs
            all_files = os.listdir(distribution_path)
            all_files = np.random.choice(
                all_files, size=min(n, len(all_files)), replace=False)
            paths = [os.path.join(distribution_path, f) for f in all_files]
            self.dummys = transform_img_fn(paths)

    def get_sample_fn(self, image, classifier_fn, lime=False):
        import copy
        # segments = slic(image, n_segments=100, compactness=20)
        segments = self.segmentation(image)
        fudged_image = image.copy()
        for x in np.unique(segments):
            fudged_image[segments == x] = (np.mean(image[segments == x][:, 0]),
                                           np.mean(image[segments == x][:, 1]),
                                           np.mean(image[segments == x][:, 2]))
        if self.white is not None:
            fudged_image[:] = self.white
        features = list(np.unique(segments))
        n_features = len(features)

        true_label = np.argmax(classifier_fn(np.expand_dims(image, 0))[0])
        print ('True pred', true_label)

        def lime_sample_fn(num_samples, batch_size=50):
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            data = np.zeros((num_samples, n_features))
            labels = []
            imgs = []
            sizes = np.random.randint(0, n_features, num_samples)
            all_features = range(n_features)
            # for row in data:
            for i, size in enumerate(sizes):
                row = np.ones(n_features)
                chosen = np.random.choice(all_features, size)
                # print chosen, size,
                row[chosen] = 0
                data[i] = row
                # print row
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
                if len(imgs) == batch_size:
                    preds = classifier_fn(np.array(imgs))
                    labels.extend(preds)
                    imgs = []
            if len(imgs) > 0:
                preds = classifier_fn(np.array(imgs))
                labels.extend(preds)
            # return imgs, np.array(labels)
            return data, np.array(labels)

        if lime:
            return segments, lime_sample_fn



        def sample_fn_dummy(present, num_samples, compute_labels=True):
            if not compute_labels:
                data = np.random.randint(
                    0, 2, num_samples * n_features).reshape(
                        (num_samples, n_features))
                data[:, present] = 1
                return [], data, []
            data = np.zeros((num_samples, n_features))
            # data = np.random.randint(0, 2, num_samples * n_features).reshape(
            #     (num_samples, n_features))
            if len(present) < 5:
                data = np.random.choice(
                    [0, 1], num_samples * n_features, p=[.8, .2]).reshape(
                        (num_samples, n_features))
            data[:, present] = 1
            chosen = np.random.choice(range(len(self.dummys)), data.shape[0],
                                      replace=True)
            labels = []
            imgs = []
            for d, r in zip(data, chosen):
                temp = copy.deepcopy(image)
                zeros = np.where(d == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                if self.white:
                    temp[mask] = 1
                else:
                    temp[mask] = self.dummys[r][mask]
                imgs.append(temp)
                # pred = np.argmax(classifier_fn(temp.to_nn())[0])
                # print self.class_names[pred]
                # labels.append(int(pred == true_label))
            # import time
            # a = time.time()
            imgs = np.array(imgs)
            preds = classifier_fn(imgs)
            # print (time.time() - a) / preds.shape[0]
            imgs = []
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            raw_data = np.hstack((data, chosen.reshape(-1, 1)))
            return raw_data, data, np.array(labels)

        def sample_fn(present, num_samples, compute_labels=True):
            # TODO: I'm sampling in this different way because the way we were
            # sampling confounds size of the document with feature presence
            # (larger documents are more likely to have features present)
            data = np.random.randint(0, 2, num_samples * n_features).reshape(
                (num_samples, n_features))
            data[:, present] = 1
            if not compute_labels:
                return [], data, []
            imgs = []
            for row in data:
                temp = copy.deepcopy(image)
                zeros = np.where(row == 0)[0]
                mask = np.zeros(segments.shape).astype(bool)
                for z in zeros:
                    mask[segments == z] = True
                temp[mask] = fudged_image[mask]
                imgs.append(temp)
            preds = classifier_fn(np.array(imgs))
            preds_max = np.argmax(preds, axis=1)
            labels = (preds_max == true_label).astype(int)
            # raw_data = imgs
            raw_data = data
            return raw_data, data, labels

        sample = sample_fn if self.hide else sample_fn_dummy
        return segments, sample

    def explain_instance(self, image, classifier_fn, threshold=0.95,
                          delta=0.1, tau=0.15, batch_size=100,
                           **kwargs):
        # classifier_fn is a predict_proba
        segments, sample = self.get_sample_fn(image, classifier_fn)
        exp = anchor_base.AnchorBaseBeam.anchor_beam(
            sample, delta=delta, epsilon=tau, batch_size=batch_size,
            desired_confidence=threshold, **kwargs)
        return segments, self.get_exp_from_hoeffding(image, exp)

    def get_exp_from_hoeffding(self, image, hoeffding_exp):
        """
        bla
        """
        ret = []

        features = hoeffding_exp['feature']
        means = hoeffding_exp['mean']
        if 'negatives' not in hoeffding_exp:
            negatives_ = [np.array([]) for x in features]
        else:
            negatives_ = hoeffding_exp['negatives']
        for f, mean, negatives in zip(features, means, negatives_):
            train_support = 0
            name = ''
            if negatives.shape[0] > 0:
                unique_negatives = np.vstack({
                    tuple(row) for row in negatives})
                distances = sklearn.metrics.pairwise_distances(
                    np.ones((1, negatives.shape[1])),
                    unique_negatives)
                negative_arrays = (unique_negatives
                                   [np.argsort(distances)[0][:4]])
                negatives = []
                for n in negative_arrays:
                    negatives.append(n)
            else:
                negatives = []
            ret.append((f, name, mean, negatives, train_support))
        return ret
