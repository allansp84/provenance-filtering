# -*- coding: utf-8 -*-

import numpy as np
from scipy import interp
import matplotlib.pyplot as plt


def apk(predicted, k=10, pos_label=1):
    """
    Computes the average precision at k.

    This function computes the average precision at k between two lists of
    items.
    """
    if len(predicted) > k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p == pos_label:
            num_hits += 1.0
            score += (num_hits / (i+1.0))

    if not predicted or not num_hits:
        return 0.0

    return score/num_hits


def mapk(predicted, k=10):
    return np.mean([apk(p, k) for p in predicted])


def precision_recall(negatives, positives, threshold):

    total_positives = float(len(positives))

    if negatives.size != 0:
        fp = negatives[negatives >= threshold].sum()
    else:
        fp = 1.

    if positives.size != 0:
        tp = positives[positives >= threshold].sum()
    else:
        tp = 1.

    if not (fp + tp):
        precision = 0.
    else:
        precision = tp / (fp + tp)

    if not total_positives:
        recall = 0.
    else:
        recall = tp / total_positives

    return precision, recall


def recall_threshold(negatives, positives, recall_value, n_points=100):

    lower_bound = min(np.min(negatives), np.min(positives))
    upper_bound = max(np.max(negatives), np.max(positives))

    steps = float((upper_bound - lower_bound)/(n_points-1))

    threshold = lower_bound
    precision, recall = 0, 1
    while recall > recall_value:

        precision, recall = precision_recall(negatives, positives, threshold)

        threshold += steps

    return precision, recall, threshold


def compute_precision_recall_curve(y_true, scores, pos_label=1, n_points=100):

    # data = np.array([scores, y_true]).T
    # sorted_idxs = np.argsort(data[:, 0])
    # scores, y_true = data[sorted_idxs, 0], data[sorted_idxs, 1]

    pos_idxs = np.where(y_true == pos_label)[0]
    neg_idxs = np.where(y_true != pos_label)[0]

    negatives = scores[neg_idxs]
    positives = scores[pos_idxs]

    if negatives.size == 0:
        neg_min, neg_max = 0., 0.
    else:
        neg_min, neg_max = np.min(negatives), np.max(negatives)

    if positives.size == 0:
        pos_min, pos_max = 0., 0.
    else:
        pos_min, pos_max = np.min(positives), np.max(positives)

    lower_bound = min(neg_min, pos_min)
    upper_bound = max(neg_max, pos_max)

    steps = float((upper_bound - lower_bound)/(n_points-1))

    threshold = lower_bound
    precisions, recalls, thresholds = [], [], []

    for pt in range(n_points):

        precision, recall = precision_recall(negatives, positives, threshold)

        precisions.append(precision)
        recalls.append(recall)

        thresholds += [threshold]
        threshold += steps

    # -- pseud point to compute the curve
    precisions.append(1)
    recalls.append(0)

    precisions = np.array(precisions)
    recalls = np.array(recalls)

    return precisions, recalls, thresholds


def compute_mean_precision_recall_curve(recalls, precisions, n_points=100):

    mean_recall = np.linspace(0, 1, n_points)
    mean_precision = 0.0
    for precision, recall in zip(precisions, recalls):
        mean_precision += interp(mean_recall, recall[::-1], precision[::-1])

    mean_precision /= len(precisions)
    mean_precision[0] = 1

    return mean_recall, mean_precision


def predicted_labels_threshold(scores, threshold, label_neg=-1, label_pos=1):

    scores = np.array(scores)

    predicted_labels = np.array([label_pos if score >= threshold else label_neg for score in scores])

    return predicted_labels


def acc_threshold(negatives, positives, threshold):

    total_positives = float(len(positives))
    total_negatives = float(len(negatives))

    if negatives.size != 0:
        tn = (np.array(negatives) < threshold).sum()
    else:
        tn = 1.

    if positives.size != 0:
        tp = (np.array(positives) >= threshold).sum()
    else:
        tp = 1.

    acc = (tp + tn)/(total_positives + total_negatives)
    return acc


def farfrr(negatives, positives, threshold):

    if negatives.size != 0:
        far = (np.array(negatives) >= threshold).mean()
    else:
        far = 1.

    if positives.size != 0:
        frr = (np.array(positives) < threshold).mean()
    else:
        frr = 1.

    return far, frr


def eer_threshold(negatives, positives, n_points=100):

    threshold = 0.0
    delta_min = 1e5

    if negatives.size == 0:
        return 0.

    if positives.size == 0:
        return 0.

    # --  we suppose that negative scores is lower than 0 and positive scores os upper than 0
    lower_bound = min(np.min(negatives), np.min(positives))
    upper_bound = max(np.max(negatives), np.max(positives))

    steps = float((upper_bound - lower_bound)/(n_points-1))

    thr = lower_bound

    for pt in range(n_points):

        far, frr = farfrr(negatives, positives, thr)

        if abs(far - frr) < delta_min:
            delta_min = abs(far - frr)
            threshold = thr

        thr += steps

    return threshold


def calc_hter(neg_devel, pos_devel, neg_test, pos_test):

    # calculate threshold upon eer point
    # threshold = bob.measure.eer_threshold(neg_devel, pos_devel)
    threshold = eer_threshold(neg_devel, pos_devel)

    # calculate far and frr
    # far, frr = bob.measure.farfrr(neg_test, pos_test, threshold)
    far, frr = farfrr(neg_test, pos_test, threshold)

    far *= 100.
    frr *= 100.

    hter = ((far + frr) / 2.)

    return threshold, far, frr, hter


def ppndf_over_array(cum_prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    n_rows, n_cols = cum_prob.shape

    norm_dev = np.zeros((n_rows, n_cols))
    for irow in range(n_rows):
        for icol in range(n_cols):

            prob = cum_prob[irow, icol]
            if prob >= 1.0:
                prob = 1-eps
            elif prob <= 0.0:
                prob = eps

            q = prob - 0.5
            if abs(prob-0.5) <= split:
                r = q * q
                pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
                pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

            else:
                if q > 0.0:
                    r = 1.0-prob
                else:
                    r = prob

                r = np.sqrt((-1.0) * np.log(r))
                pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
                pf /= ((d_2 * r + d_1) * r + 1.0)

                if q < 0:
                    pf *= -1.0

            norm_dev[irow, icol] = pf

    return norm_dev


def ppndf(prob):
    split = 0.42
    a_0 = 2.5066282388
    a_1 = -18.6150006252
    a_2 = 41.3911977353
    a_3 = -25.4410604963
    b_1 = -8.4735109309
    b_2 = 23.0833674374
    b_3 = -21.0622410182
    b_4 = 3.1308290983
    c_0 = -2.7871893113
    c_1 = -2.2979647913
    c_2 = 4.8501412713
    c_3 = 2.3212127685
    d_1 = 3.5438892476
    d_2 = 1.6370678189
    eps = 2.2204e-16

    if prob >= 1.0:
        prob = 1-eps
    elif prob <= 0.0:
        prob = eps

    q = prob - 0.5
    if abs(prob-0.5) <= split:
        r = q * q
        pf = q * (((a_3 * r + a_2) * r + a_1) * r + a_0)
        pf /= (((b_4 * r + b_3) * r + b_2) * r + b_1) * r + 1.0

    else:
        if q > 0.0:
            r = 1.0-prob
        else:
            r = prob

        r = np.sqrt((-1.0) * np.log(r))
        pf = (((c_3 * r + c_2) * r + c_1) * r + c_0)
        pf /= ((d_2 * r + d_1) * r + 1.0)

        if q < 0:
            pf *= -1.0

    return pf


def compute_det(negatives, positives, npoints):

    lower_bound = min(np.min(negatives), np.min(negatives))
    upper_bound = max(np.max(negatives), np.max(negatives))

    steps = float((upper_bound - lower_bound)/(npoints-1))

    threshold = lower_bound
    curve = []
    for pt in range(npoints):

        far, frr = farfrr(negatives, positives, threshold)

        curve.append([far, frr])
        threshold += steps

    curve = np.array(curve)

    return ppndf_over_array(curve.T)


def det(negatives, positives, n_points, axis_font_size='x-small', **kwargs):

    # these are some constants required in this method
    desired_ticks = [
        '0.00001', '0.00002', '0.00005',
        '0.0001', '0.0002', '0.0005',
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.4', '0.6', '0.8', '0.9',
        '0.95', '0.98', '0.99',
        '0.995', '0.998', '0.999',
        '0.9995', '0.9998', '0.9999',
        '0.99995', '0.99998', '0.99999',
    ]

    desired_labels = [
        '0.001', '0.002', '0.005',
        '0.01', '0.02', '0.05',
        '0.1', '0.2', '0.5',
        '1', '2', '5',
        '10', '20', '40', '60', '80', '90',
        '95', '98', '99',
        '99.5', '99.8', '99.9',
        '99.95', '99.98', '99.99',
        '99.995', '99.998', '99.999',
    ]

    curve = compute_det(negatives, positives, n_points)

    output_plot = plt.plot(curve[0, :], curve[1, :], **kwargs)

    # -- now the trick: we must plot the tick marks by hand using the PPNDF method
    p_ticks = [ppndf(float(v)) for v in desired_ticks]

    # -- and finally we set our own tick marks
    ax = plt.gca()
    ax.set_xticks(p_ticks)
    ax.set_xticklabels(desired_labels, size=axis_font_size)
    ax.set_yticks(p_ticks)
    ax.set_yticklabels(desired_labels, size=axis_font_size)

    return output_plot


def det_axis(v, **kwargs):

    tv = list(v)
    tv = [ppndf(float(k)/100) for k in tv]
    ret = plt.axis(tv, **kwargs)

    return ret


def split_score_distributions(gt, predicted_scores, pos_label=1):
    # -- get the score distributions of positive and negative classes

    pos_idxs = gt == pos_label
    neg_idxs = gt != pos_label

    pos = predicted_scores[pos_idxs]
    neg = predicted_scores[neg_idxs]

    # pos = [score for label, score in zip(gt, predicted_scores) if label == pos_label]
    # neg = [score for label, score in zip(gt, predicted_scores) if label != pos_label]
    # return np.array(neg), np.array(pos)

    return neg, pos
