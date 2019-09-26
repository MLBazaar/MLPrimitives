# -*- coding: utf-8 -*-

import logging
from functools import partial

import keras
import numpy as np
import pandas as pd
from keras import backend as K
from keras.layers import (
    LSTM, Activation, Bidirectional, Conv1D, Dense, Dropout, Flatten, Input, Reshape,
    TimeDistributed)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D
from keras.layers.merge import _Merge
from keras.models import Model
from keras.optimizers import Adam
from scipy import integrate, stats

LOGGER = logging.getLogger(__name__)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class CycleGAN():
    """CycleGAN class"""

    def build_encoder(self):
        x = Input(shape=self.shape)
        model = keras.models.Sequential()
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        model.add(Flatten())
        model.add(Dense(self.latent_dim))
        model.add(Reshape((self.latent_dim, 1)))
        z = model(x)
        return Model(x, z)

    def build_generator(self):
        z = Input(shape=(self.latent_dim, 1))
        model = keras.models.Sequential()
        model.add(Flatten())
        model.add(Dense(50))
        model.add(Reshape((50, 1)))
        model.add(
            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2),
                merge_mode='concat'))
        model.add(UpSampling1D(2))
        model.add(
            Bidirectional(
                LSTM(
                    64,
                    return_sequences=True,
                    dropout=0.2,
                    recurrent_dropout=0.2),
                merge_mode='concat'))
        model.add(TimeDistributed(Dense(1)))
        model.add(Activation("tanh"))
        x_ = model(z)
        return Model(z, x_)

    def build_critic_x(self):
        x = Input(shape=(self.shape[0], 1))
        model = keras.models.Sequential()
        model.add(Conv1D(64, 5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, 5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, 5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv1D(64, 5))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1))
        validity = model(x)
        return Model(x, validity)

    def build_critic_z(self):
        z = Input(shape=(self.latent_dim, 1))
        model = keras.models.Sequential()
        model.add(Flatten())
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(100))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        validity = model(z)
        return Model(z, validity)

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        gradients = K.gradients(y_pred, averaged_samples)[0]
        gradients_sqr = K.square(gradients)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = K.square(1 - gradient_l2_norm)
        return K.mean(gradient_penalty)

    def __init__(self, epochs=2000, shape=(100, 1), latent_dim=20, batch_size=64, n_critic=5):
        """Initialize the ARIMA object.

        Args:
            epochs (int):
                Optional. Integer denoting the number of epochs.
            shape (tuple):
                Optional. Tuple denoting the shape of an input sample.
            latent_dim (int):
                Optional. Integer denoting dimension of latent space.
            batch_size (int):
                Integer denoting the batch size.
            n_critic (int):
                Integer denoting the number of critic training steps per one
                Generator/Encoder training step.
        """

        self.shape = shape
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.epochs = epochs

        optimizer = Adam(lr=0.0005)

        self.encoder = self.build_encoder()
        self.generator = self.build_generator()

        self.critic_x = self.build_critic_x()
        self.critic_z = self.build_critic_z()

        self.generator.trainable = False
        self.encoder.trainable = False

        z = Input(shape=(self.latent_dim, 1))
        x = Input(shape=self.shape)
        x_ = self.generator(z)
        z_ = self.encoder(x)
        fake_x = self.critic_x(x_)
        valid_x = self.critic_x(x)
        interpolated_x = RandomWeightedAverage()([x, x_])

        validity_interpolated_x = self.critic_x(interpolated_x)
        partial_gp_loss_x = partial(self.gradient_penalty_loss, averaged_samples=interpolated_x)
        partial_gp_loss_x.__name__ = 'gradient_penalty'
        self.critic_x_model = Model(
            inputs=[
                x, z], outputs=[
                valid_x, fake_x, validity_interpolated_x])
        self.critic_x_model.compile(
            loss=[
                self.wasserstein_loss,
                self.wasserstein_loss,
                partial_gp_loss_x],
            optimizer=optimizer,
            loss_weights=[
                1,
                1,
                5])

        fake_z = self.critic_z(z_)
        valid_z = self.critic_z(z)
        interpolated_z = RandomWeightedAverage()([z, z_])
        validity_interpolated_z = self.critic_z(interpolated_z)
        partial_gp_loss_z = partial(self.gradient_penalty_loss, averaged_samples=interpolated_z)
        partial_gp_loss_z.__name__ = 'gradient_penalty'
        self.critic_z_model = Model(
            inputs=[
                x, z], outputs=[
                valid_z, fake_z, validity_interpolated_z])
        self.critic_z_model.compile(
            loss=[
                self.wasserstein_loss,
                self.wasserstein_loss,
                partial_gp_loss_z],
            optimizer=optimizer,
            loss_weights=[
                1,
                1,
                10])

        self.critic_x.trainable = False
        self.critic_z.trainable = False
        self.generator.trainable = True
        self.encoder.trainable = True

        z_gen = Input(shape=(self.latent_dim, 1))
        x_gen_ = self.generator(z_gen)
        x_gen = Input(shape=self.shape)
        z_gen_ = self.encoder(x_gen)
        x_gen_rec = self.generator(z_gen_)
        fake_gen_x = self.critic_x(x_gen_)
        fake_gen_z = self.critic_z(z_gen_)

        self.generator_model = Model([x_gen, z_gen], [fake_gen_x, fake_gen_z, x_gen_rec])
        self.generator_model.compile(
            loss=[
                self.wasserstein_loss,
                self.wasserstein_loss,
                'mse'],
            optimizer=optimizer,
            loss_weights=[
                1,
                1,
                50])

    def train(self, X, epochs):
        X_train = X
        fake = np.ones((self.batch_size, 1))
        valid = -np.ones((self.batch_size, 1))
        delta = np.ones((self.batch_size, 1)) * 10

        for epoch in range(epochs):
            for _ in range(self.n_critic):
                idx = np.random.randint(0, X_train.shape[0], self.batch_size)
                x = X_train[idx]
                z = np.random.normal(size=(self.batch_size, self.latent_dim, 1))

                cx_loss = self.critic_x_model.train_on_batch([x, z], [valid, fake, delta])
                cz_loss = self.critic_z_model.train_on_batch([x, z], [valid, fake, delta])

            g_loss = self.generator_model.train_on_batch([x, z], [valid, valid, x])

            if epoch % 100 == 0:
                print(
                    "Epoch:",
                    epoch,
                    "[Dx loss: ",
                    cx_loss,
                    "] [Dz loss: ",
                    cz_loss,
                    "] [G loss: ",
                    g_loss,
                    "]")

    def fit(self, X):
        """Fit the CycleGAN.

        Args:
            X (ndarray):
                N-dimensional array containing the input training sequences for the model.
        """
        X = X.reshape((-1, self.shape[0], 1))
        self.train(X, epochs=self.epochs)

    def predict(self, X):
        """Predict values using the initialized object.

        Args:
            X (ndarray):
                N-dimensional array containing the input sequences for the model.

        Returns:
            ndarray:
                N-dimensional array containing the reconstructions for each input sequence.
            ndarray:
                N-dimensional array containing the critic scores for each input sequence.
        """
        X = X.reshape((-1, self.shape[0], 1))
        z_ = self.encoder.predict(X)
        y_hat = self.generator.predict(z_)
        critic = self.critic_x.predict(X)

        return y_hat, critic


def score_anomalies(y, y_hat, critic, score_window=10, smooth_window=200):
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
