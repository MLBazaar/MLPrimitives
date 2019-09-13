import numpy as np
import pandas as pd
from scipy import integrate, stats


def errors_cyclegan(y, y_hat, critic, score_window=10, smooth_window=200):
    """Compute an array of error scores.

    Errors are calculated using a combination of reconstruction error and critic score.

    Args:
        y (ndarray):
            Ground truth.
        y_hat (ndarray):
            Predicted values. Each timestamp has multiple predictions.
        critic (ndarray):
            Critic score. Each timestamp has multiple critic scores.
        score_window (int):
            Optional. Size of the window over which the scores are calculated.
            If not given, 10 is used.
        smooth_window (int):
            Optional. Size of window over which smoothing is applied.
            If not given, 200 is used.

    Returns:
        ndarray:
            Array of errors.
    """

    true = [item[0] for item in y.reshape((y.shape[0], -1))]

    for item in y[-1][1:]:
        true.extend(item)

    critic_extended = list()
    for c in critic:
        critic_extended = critic_extended + np.repeat(c, y_hat.shape[1]).tolist()

    step_size = 1
    predictions = []
    critic_kde_max = []
    pred_length = y_hat.shape[1]
    num_errors = y_hat.shape[1] + step_size * (y_hat.shape[0] - 1)
    y_hat = np.asarray(y_hat)
    critic_extended = np.asarray(critic_extended).reshape((-1, y_hat.shape[1]))

    for i in range(num_errors):
        intermediate = []
        critic_intermediate = []
        for j in range(max(0, i - num_errors + pred_length), min(i + 1, pred_length)):
            intermediate.append(y_hat[i - j, j])
            critic_intermediate.append(critic_extended[i - j, j])
        if intermediate:
            predictions.append(np.median(np.asarray(intermediate)))
            if len(critic_intermediate) > 1:
                discr_intermediate = np.asarray(critic_intermediate)
                try:
                    critic_kde_max.append(discr_intermediate[np.argmax(
                        stats.gaussian_kde(discr_intermediate)(critic_intermediate))])
                except np.linalg.LinAlgError:
                    critic_kde_max.append(np.median(discr_intermediate))
                    continue
            else:
                critic_kde_max.append(np.median(np.asarray(critic_intermediate)))

    predictions = np.asarray(predictions)

    score_window_min = int(score_window / 2)

    scores = pd.Series(
        abs(
            pd.Series(np.asarray(true).flatten()).rolling(
                score_window, center=True,
                min_periods=score_window_min
            ).apply(
                integrate.trapz
            ) - pd.Series(
                np.asarray(predictions).flatten()
            ).rolling(
                score_window,
                center=True,
                min_periods=score_window_min
            ).apply(
                integrate.trapz
            )
        )
    ).rolling(
        smooth_window, center=True, min_periods=int(smooth_window / 2),
        win_type='triang').mean().values

    z_score_scores = stats.zscore(scores)

    critic_kde_max = np.asarray(critic_kde_max)
    l_quantile = np.quantile(critic_kde_max, 0.25)
    u_quantile = np.quantile(critic_kde_max, 0.75)
    in_range = np.logical_and(critic_kde_max >= l_quantile, critic_kde_max <= u_quantile)
    critic_mean = np.mean(critic_kde_max[in_range])
    critic_std = np.std(critic_kde_max)

    z_score_critic = np.absolute((np.asarray(critic_kde_max) - critic_mean) / critic_std) + 1
    z_score_critic = pd.Series(z_score_critic).rolling(
        100, center=True, min_periods=50).mean().values
    z_score_scores_clip = np.clip(z_score_scores, a_min=0, a_max=None) + 1

    multiply_comb = np.multiply(z_score_scores_clip, z_score_critic)

    return multiply_comb
