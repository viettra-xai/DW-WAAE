import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import LeakyReLU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import pandas as pd
import os
import sys
import random
import time

class DW_WAAE:
    def __init__(self, input_dim, latent_dim=50, batch_size=64, critic_iterations=5, lambda_gp=10.0):
        # Hyperparameters
        self.input_dim = input_dim  # Input dimension passed as an external parameter
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.critic_iterations = critic_iterations
        self.lambda_gp = lambda_gp

        # Build models
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = self.build_autoencoder()
        self.critic = self.build_critic()

        # Optimizers
        self.autoencoder_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.critic_optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.00005)

    def build_encoder(self):
        """Builds the encoder model."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.input_dim,)),
            layers.Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(128, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(self.latent_dim, activation="linear", kernel_initializer="lecun_normal")  # Encodes to latent space
        ])
        return model

    def build_decoder(self):
        """Builds the decoder model."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.latent_dim,)),
            layers.Dense(128, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(self.input_dim, activation="sigmoid", kernel_initializer="lecun_normal")  # Outputs reconstructed data
        ])
        return model

    def build_autoencoder(self):
        """Builds the autoencoder model by combining encoder and decoder."""
        inputs = layers.Input(shape=(self.input_dim,))
        latent_space = self.encoder(inputs)
        outputs = self.decoder(latent_space)
        return tf.keras.Model(inputs, outputs)

    def build_critic(self):
        """Builds the critic (discriminator) model."""
        model = tf.keras.Sequential([
            layers.Input(shape=(self.input_dim * 2,)),  # Input shape is doubled for stacked (x, x) or (x, x_hat)
            layers.Dense(256, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(128, activation="selu", kernel_initializer="lecun_normal"),
            layers.Dense(1, kernel_initializer="lecun_normal")  # Linear output for Wasserstein distance
        ])
        return model

    def gradient_penalty(self, real_images, fake_images):
        """Computes the gradient penalty for WGAN-GP."""
        batch_size = tf.shape(real_images)[0]
        alpha = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interpolated = alpha * real_images + (1 - alpha) * fake_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            interpolated_output = self.critic(interpolated, training=True)
        grads = tape.gradient(interpolated_output, [interpolated])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1))
        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    def critic_loss(self, real_output, fake_output, anomaly_scores):
        """Critic loss function incorporating outlier scores."""

        anomaly_scores = tf.sigmoid(50 * (anomaly_scores - 0.5))
        diff = tf.nn.relu(real_output - fake_output) + 1e-6  # Clamp and add a small epsilon to avoid zero
        term1 = (1 - anomaly_scores) * diff
        term2 = anomaly_scores * tf.math.log(1 - tf.math.exp(-diff))
        
        return -tf.reduce_mean(term1 + term2)

    def autoencoder_loss(self, real_output, fake_output, anomaly_scores):
        """Loss function for encoder-decoder (autoencoder) incorporating outlier scores. """
        anomaly_scores = tf.sigmoid(50 * (anomaly_scores - 0.5))
        diff = tf.nn.relu(real_output - fake_output) + 1e-6  # Clamp and add a small epsilon to avoid zero
        term1 = (1 - anomaly_scores) * diff
        term2 = -anomaly_scores * tf.math.log(1 - tf.math.exp(-diff))
  
        return tf.reduce_mean(term1 + term2)


    def monitor_loss(self, x, anomaly_scores):

        diff = self.compute_r_score(x)

        term1 = (1 - anomaly_scores) * diff
        term2 = -anomaly_scores * tf.math.log(1 - tf.math.exp(-diff))
  
        return tf.reduce_mean(term1 + term2)

    def calculate_monitor_loss(self, data, anomaly_scores, high_percentile=95, low_percentile=5):
 
        # Filter data and anomaly scores for top percentiles
        high_score_threshold = np.percentile(anomaly_scores, high_percentile)
        high_score_indices = np.where(anomaly_scores >= high_score_threshold)[0]
        high_score_data = data.numpy()[high_score_indices]
        high_score_labels = np.ones(len(high_score_indices))  # Label = 1 for high anomaly scores
    
        # Filter data and anomaly scores for bottom percentiles
        low_score_threshold = np.percentile(anomaly_scores, low_percentile)
        low_score_indices = np.where(anomaly_scores <= low_score_threshold)[0]
        low_score_data = data.numpy()[low_score_indices]
        low_score_labels = np.zeros(len(low_score_indices))  # Label = 0 for low anomaly scores
    
        # Combine the high and low score data and labels
        combined_data = np.concatenate([high_score_data, low_score_data], axis=0)
        combined_labels = np.concatenate([high_score_labels, low_score_labels], axis=0)
    
        # Convert to TensorFlow tensors
        combined_data = tf.convert_to_tensor(combined_data, dtype=tf.float32)
        combined_labels = tf.convert_to_tensor(combined_labels, dtype=tf.float32)
    
        # Calculate and return monitor loss
        monitor_loss = self.monitor_loss(combined_data, combined_labels).numpy()
        return monitor_loss



    def get_penultimate_layer_output(self, inputs):
        """Returns the output of the penultimate layer of the critic."""
        # Define the input shape explicitly (adjust the shape as per your model)
        input_tensor = tf.keras.Input(shape=(self.input_dim * 2,))  # Define input shape
    
        # Extract penultimate layer output by rebuilding the sub-model
        x = input_tensor
        for layer in self.critic.layers[:-1]:  # Exclude the final output layer
            x = layer(x)
        
        # Create the sub-model
        penultimate_model = tf.keras.Model(inputs=input_tensor, outputs=x)
        return penultimate_model(inputs)


    def compute_r_score(self, x):
        """Computes the R-score based on the penultimate layer's feature space."""
        # Reconstruct the input using the autoencoder
        x_reconstructed = self.autoencoder(x, training=False)
        
        # Create stacked (x, x) and (x, x_reconstructed) for the critic
        real_stacked = tf.concat([x, x], axis=1)
        fake_stacked = tf.concat([x, x_reconstructed], axis=1)
        
        # Get penultimate layer outputs for both real and reconstructed data
        f_real = self.get_penultimate_layer_output(real_stacked)
        f_fake = self.get_penultimate_layer_output(fake_stacked)
        
        # Compute the R-score using L1 norm
        r_score = tf.reduce_mean(tf.abs(f_real - f_fake), axis=1)  # Per-sample R-score
        return r_score


    @tf.function
    def train_step(self, real_data, soft_anomaly_scores):
        # Train the critic
        for _ in range(self.critic_iterations):
            with tf.GradientTape() as tape:
                fake_data = self.autoencoder(real_data, training=True)
                real_stacked = tf.concat([real_data, real_data], axis=1)
                fake_stacked = tf.concat([real_data, fake_data], axis=1)
                real_output = self.critic(real_stacked, training=True)
                fake_output = self.critic(fake_stacked, training=True)
                gp = self.gradient_penalty(real_stacked, fake_stacked)
                c_loss = self.critic_loss(real_output, fake_output, soft_anomaly_scores) + self.lambda_gp * gp

            gradients_of_critic = tape.gradient(c_loss, self.critic.trainable_variables)
            clipped_gradients = [tf.clip_by_value(grad, -0.01, 0.01) for grad in gradients_of_critic]
            self.critic_optimizer.apply_gradients(zip(clipped_gradients, self.critic.trainable_variables))

        # Train the autoencoder
        with tf.GradientTape() as tape:
            fake_data = self.autoencoder(real_data, training=True)
            real_stacked = tf.concat([real_data, real_data], axis=1)
            fake_stacked = tf.concat([real_data, fake_data], axis=1)
            real_output = self.critic(real_stacked, training=True)
            fake_output = self.critic(fake_stacked, training=True)
            ae_loss = self.autoencoder_loss(real_output, fake_output, soft_anomaly_scores)

        gradients_of_autoencoder = tape.gradient(ae_loss, self.autoencoder.trainable_variables)
        self.autoencoder_optimizer.apply_gradients(zip(gradients_of_autoencoder, self.autoencoder.trainable_variables))

        return c_loss, ae_loss


    def train(self, data, X_test, y_test, initial_epochs=10, epochs=100, patience=3):
        BUFFER_SIZE = data.shape[0]

        # Convert data to TensorFlow tensor
        data = tf.convert_to_tensor(data, dtype=tf.float32)

        # Define paths for saving model weights
        best_autoencoder_path = "models/best_autoencoder.weights.h5"
        best_critic_path = "models/best_critic.weights.h5"

    
        # # Initialize anomaly_scores as an array of zeros

        anomaly_scores = tf.random.uniform(shape=(data.shape[0],), minval=0.0, maxval=0.5, dtype=tf.float32)

        # Combine data and initial anomaly scores into a dataset
        dataset = tf.data.Dataset.from_tensor_slices((data, anomaly_scores))
        dataset = dataset.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).prefetch(1)

        anomaly_scores = anomaly_scores.numpy()

        # Variables to track monitor loss and decrease coefficients
        monitor_losses = []  # To store monitor losses for each epoch
        decrease_coefficients = [0]  # Initialize with the first item as zero

         # Variables for early stopping
        best_monitor_loss = float("inf")
        patience_counter = 0

        for epoch in range(initial_epochs, epochs):
            epoch_start = time.time()
            batch_c_losses, batch_ae_losses = [], []

            # Iterate over all batches for continued training
            for real_data, batch_anomaly_scores in dataset:
                c_loss, ae_loss = self.train_step(real_data, batch_anomaly_scores)
    
                # Accumulate batch losses
                batch_c_losses.append(c_loss.numpy())
                batch_ae_losses.append(ae_loss.numpy())
    
            # Calculate average losses over the entire dataset
            avg_c_loss = np.mean(batch_c_losses)
            avg_ae_loss = np.mean(batch_ae_losses)

            monitor_loss = self.calculate_monitor_loss(data, anomaly_scores)
    
            # Append monitor loss to list
            monitor_losses.append(monitor_loss)
    
            # Calculate decrease coefficient if more than one epoch
            if len(monitor_losses) > 1:
                decrease = (monitor_losses[-2] - monitor_losses[-1]) / max(monitor_losses[-2], 1e-8)
                decrease_coefficients.append(decrease)
    
            if X_test is not None and y_test is not None:
                # Compute R-score for X_test
                y_pred = self.compute_r_score(X_test).numpy()
        
                # Set anomaly detection threshold (e.g., 50th percentile)
                anomaly_threshold = np.percentile(y_pred, 50)
        
                # Predict anomalies
                y_pred_flag = np.where(y_pred >= anomaly_threshold, 1, 0)
        
                # Calculate precision, recall, and F1-score
                prec, recall, fscore, _ = precision_recall_fscore_support(y_test, y_pred_flag, average="binary")
        
                # Display losses and F1 Score metrics
                elapsed_time = time.time() - epoch_start
                print(f"Epoch {epoch + 1}/{epochs} - Time: {elapsed_time:.2f}s")
                print(f"Critic Loss: {avg_c_loss:.4f} - Autoencoder Loss: {avg_ae_loss:.4f} - Monitor Loss: {monitor_loss:.4f} - Decrease Coefficient: {decrease_coefficients[-1]:.4f}")
                print(f" Precision = {prec:.3f}")
                print(f" Recall    = {recall:.3f}")
                print(f" F1-Score  = {fscore:.3f}")

            # Early stopping and saving the best model
            if epoch + 1 > 20:  # Start monitoring after epoch 20
                if monitor_loss < best_monitor_loss:
                    best_monitor_loss = monitor_loss
                    patience_counter = 0
    
                    # Save the best models
                    self.autoencoder.save_weights(best_autoencoder_path)
                    self.critic.save_weights(best_critic_path)
                    print(f"Saved best models at epoch {epoch + 1}")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1} due to no improvement in monitor_loss.")
                        break


            # Compute new anomaly scores using the critic for next epoch
            # print("Estimating anomaly scores using critic...")
            anomaly_scores = self.compute_r_score(data).numpy()
        
            # Normalize anomaly scores to [0, 1]
            score_scaler = MinMaxScaler()
            anomaly_scores = score_scaler.fit_transform(anomaly_scores.reshape(-1, 1)).flatten()
        
            # Select 70% lowest scores and 5% highest scores
            low_percentile = np.percentile(anomaly_scores, 70)
            high_percentile = np.percentile(anomaly_scores, 95)
        
            selected_indices = np.where((anomaly_scores <= low_percentile) | (anomaly_scores >= high_percentile))[0]
        
            # Filter data and anomaly scores
            data_new = data.numpy()[selected_indices]
            anomaly_scores_new = anomaly_scores[selected_indices]
        
            # Convert back to TensorFlow tensors
            data_new = tf.convert_to_tensor(data_new, dtype=tf.float32)
            anomaly_scores_new = tf.convert_to_tensor(anomaly_scores_new, dtype=tf.float32)
        
            # Create a new dataset with filtered data and scores
            dataset = tf.data.Dataset.from_tensor_slices((data_new, anomaly_scores_new))
            dataset = dataset.shuffle(BUFFER_SIZE).batch(self.batch_size, drop_remainder=True).prefetch(1)

        # Restore the best models for testing
        self.autoencoder.load_weights(best_autoencoder_path)
        self.critic.load_weights(best_critic_path)
        print("Best models restored for testing.")
