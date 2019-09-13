# -*- coding: utf-8 -*-

import logging
from functools import partial

import keras
import numpy as np
from keras import backend as K
from keras.layers import (
    LSTM, Activation, Bidirectional, Conv1D, Dense, Dropout, Flatten, Input, Reshape,
    TimeDistributed)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D
from keras.layers.merge import _Merge
from keras.models import Model
from keras.optimizers import Adam

LOGGER = logging.getLogger(__name__)


class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((64, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])


class LSTM_CycleGAN():
    """LSTM CycleGAN class"""

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
