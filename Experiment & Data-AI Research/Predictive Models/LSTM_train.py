import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class SoilPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_scalers = {}
        
    def clean_numeric_data(self, series):
        """Convert series to numeric, handling 'NO WIFI' and other non-numeric values"""
        return pd.to_numeric(series, errors='coerce')
        
    def load_and_preprocess_data(self, filename):
        """Load and preprocess the soil sensor data"""
        df = pd.read_csv(filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Clean all numeric columns
        numeric_columns = [col for col in df.columns if col != 'timestamp']
        for col in numeric_columns:
            df[col] = self.clean_numeric_data(df[col])
        
        # Filter out days with rain > 10 (only if daily_rain_sum exists and is numeric)
        if 'daily_rain_sum' in df.columns:
            # Ensure daily_rain_sum is numeric before filtering
            df['daily_rain_sum'] = self.clean_numeric_data(df['daily_rain_sum'])
            df = df[df['daily_rain_sum'] <= 10]
        
        # Extract time features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Fill remaining NaN values
        df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
        
        print(f"Data loaded: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def create_sequences(self, data, target_soil, sequence_length=24):
        """Create sequences for LSTM training"""
        sequences = []
        targets = []
        
        # Define target columns based on soil type
        target_columns = [f'{target_soil}_humidity', f'{target_soil}_ph', 
                         f'{target_soil}_ec', f'{target_soil}_n', 
                         f'{target_soil}_p', f'{target_soil}_k']
        
        # Feature columns (using all available sensors and environmental data)
        feature_columns = []
        for soil in ['soil', 'soil2', 'soil3']:
            feature_columns.extend([f'{soil}_temp', f'{soil}_humidity', 
                                 f'{soil}_ec', f'{soil}_ph', f'{soil}_n', 
                                 f'{soil}_p', f'{soil}_k'])
        
        # Add environmental features
        env_columns = ['air_temp', 'air_wind', 'dewpoint', 'precip_prob', 
                      'daily_temp_max', 'daily_temp_min', 'daily_rain_sum', 
                      'sunshine_duration', 'hour', 'day_of_week', 'day_of_year']
        
        feature_columns.extend([col for col in env_columns if col in data.columns])
        
        # Remove any columns that might not exist or have issues
        available_feature_columns = [col for col in feature_columns if col in data.columns]
        available_target_columns = [col for col in target_columns if col in data.columns]
        
        print(f"Using {len(available_feature_columns)} feature columns")
        print(f"Using {len(available_target_columns)} target columns: {available_target_columns}")
        
        # Ensure we have enough data
        if len(data) < sequence_length + 1:
            raise ValueError(f"Not enough data. Need at least {sequence_length + 1} rows, but only have {len(data)}")
        
        # Scale features
        feature_scaler = MinMaxScaler()
        scaled_features = feature_scaler.fit_transform(data[available_feature_columns])
        
        # Scale targets
        target_scaler = MinMaxScaler()
        scaled_targets = target_scaler.fit_transform(data[available_target_columns])
        
        # Create sequences
        for i in range(len(scaled_features) - sequence_length):
            sequences.append(scaled_features[i:i+sequence_length])
            targets.append(scaled_targets[i+sequence_length])
        
        if len(sequences) == 0:
            raise ValueError("No sequences created. Check sequence_length and data size.")
        
        print(f"Created {len(sequences)} sequences")
        return (np.array(sequences), np.array(targets), 
                feature_scaler, target_scaler, available_feature_columns, available_target_columns)
    
    def build_model(self, input_shape, output_dim):
        """Build LSTM model with optimized architecture"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            Dropout(0.3),
            LSTM(64, return_sequences=True,
                 kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            Dropout(0.3),
            LSTM(32, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal'),
            Dropout(0.3),
            Dense(64, activation='relu', kernel_initializer='he_normal'),
            Dense(32, activation='relu', kernel_initializer='he_normal'),
            Dense(output_dim, activation='linear')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001, clipnorm=1.0),
            loss='mse', 
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train_models(self, data_file, sequence_length=24, epochs=100):
        """Train models for each soil type"""
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess_data(data_file)
        print(f"Data loaded successfully: {len(df)} rows")
        
        # Define callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        for soil_type in ['soil2', 'soil3']:
            print(f"\n{'='*60}")
            print(f"Training model for {soil_type}")
            print(f"{'='*60}")
            
            try:
                # Create sequences
                X, y, feature_scaler, target_scaler, feature_cols, target_cols = self.create_sequences(
                    df, soil_type, sequence_length
                )
                
                print(f"Training data shape - X: {X.shape}, y: {y.shape}")
                
                # Split data (80% train, 20% validation)
                split_idx = int(0.8 * len(X))
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]
                
                print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
                
                # Build and train model
                model = self.build_model((X.shape[1], X.shape[2]), y.shape[1])
                
                print("\nModel architecture:")
                model.summary()
                
                print("\nStarting training...")
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=1,
                    shuffle=False
                )
                
                # Store model and scalers
                self.models[soil_type] = model
                self.scalers[soil_type] = target_scaler
                self.feature_scalers[soil_type] = feature_scaler
                
                # Plot training history
                self.plot_training_history(history, soil_type)
                
                # Calculate and display final metrics
                self.evaluate_model(model, X_train, y_train, X_val, y_val, target_scaler, soil_type)
                
                print(f"✓ Training completed successfully for {soil_type}!")
                
            except Exception as e:
                print(f"✗ Error training model for {soil_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    def plot_training_history(self, history, soil_type):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(history.history['loss'], label='Training Loss', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title(f'{soil_type} - Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # MAE
        ax2.plot(history.history['mae'], label='Training MAE', linewidth=2)
        ax2.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title(f'{soil_type} - Mean Absolute Error')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # MSE
        ax3.plot(history.history['mse'], label='Training MSE', linewidth=2)
        ax3.plot(history.history['val_mse'], label='Validation MSE', linewidth=2)
        ax3.set_title(f'{soil_type} - Mean Squared Error')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('MSE')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Learning rate (if available)
        if 'lr' in history.history:
            ax4.plot(history.history['lr'], label='Learning Rate', linewidth=2, color='red')
            ax4.set_title(f'{soil_type} - Learning Rate')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Learning Rate')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{soil_type}_training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Training plots saved as {soil_type}_training_history.png")
    
    def evaluate_model(self, model, X_train, y_train, X_val, y_val, target_scaler, soil_type):
        """Evaluate model performance"""
        # Make predictions
        train_pred = model.predict(X_train, verbose=0)
        val_pred = model.predict(X_val, verbose=0)
        
        # Inverse transform predictions
        train_pred_inv = target_scaler.inverse_transform(train_pred)
        val_pred_inv = target_scaler.inverse_transform(val_pred)
        y_train_inv = target_scaler.inverse_transform(y_train)
        y_val_inv = target_scaler.inverse_transform(y_val)
        
        # Calculate metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_inv, train_pred_inv))
        val_rmse = np.sqrt(mean_squared_error(y_val_inv, val_pred_inv))
        
        train_mae = mean_absolute_error(y_train_inv, train_pred_inv)
        val_mae = mean_absolute_error(y_val_inv, val_pred_inv)
        
        print(f"\nModel Evaluation for {soil_type}:")
        print(f"{'-'*40}")
        print(f"Training RMSE:   {train_rmse:.4f}")
        print(f"Validation RMSE: {val_rmse:.4f}")
        print(f"Training MAE:    {train_mae:.4f}")
        print(f"Validation MAE:   {val_mae:.4f}")
        
        # Calculate R² score
        from sklearn.metrics import r2_score
        train_r2 = r2_score(y_train_inv, train_pred_inv)
        val_r2 = r2_score(y_val_inv, val_pred_inv)
        print(f"Training R²:     {train_r2:.4f}")
        print(f"Validation R²:   {val_r2:.4f}")
    
    def save_models(self):
        """Save trained models using Keras native format"""
        import joblib
        import os
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        for soil_type in self.models:
            try:
                # Save model in Keras native format
                model_path = f'models/{soil_type}_model'
                self.models[soil_type].save(model_path)
                
                # Save scalers
                joblib.dump(self.scalers[soil_type], f'models/{soil_type}_target_scaler.pkl')
                joblib.dump(self.feature_scalers[soil_type], f'models/{soil_type}_feature_scaler.pkl')
                
                print(f"✓ Saved {soil_type} model to: {model_path}")
                
            except Exception as e:
                print(f"✗ Error saving model for {soil_type}: {str(e)}")
        
        print(f"\n✓ All models and scalers saved in 'models' directory")
        
        # Save model info
        model_info = {
            'trained_models': list(self.models.keys()),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open('models/training_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print("✓ Training info saved")

# Train the models
if __name__ == "__main__":
    print("Soil Sensor LSTM Model Training")
    print("=" * 50)
    
    predictor = SoilPredictor()
    
    try:
        predictor.train_models('SoilSensorLog_clean.csv', sequence_length=24, epochs=100)
        predictor.save_models()
        
        print("\n" + "=" * 50)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Trained models: {list(predictor.models.keys())}")
        
    except Exception as e:
        print(f"\n✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()