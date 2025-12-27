import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import streamlit as st

class PyTorchRNN(nn.Module):
    def __init__(self, model_type='LSTM', n_units=50, input_dim=1):
        super(PyTorchRNN, self).__init__()
        self.model_type = model_type
        self.n_units = n_units
        
        if model_type == 'LSTM':
            self.rnn = nn.LSTM(input_dim, n_units, batch_first=True)
        elif model_type == 'GRU':
            self.rnn = nn.GRU(input_dim, n_units, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, n_units, batch_first=True)
            
        self.fc = nn.Linear(n_units, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        out, _ = self.rnn(x)
        # out shape: (batch_size, seq_len, n_units)
        # Use the output of the last time step
        out = self.fc(out[:, -1, :])
        return out

class PyTorchAutoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=2):
        super(PyTorchAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoding_dim),
            nn.Identity() # Placeholder to match 'linear' activation
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, input_dim),
            nn.Identity() # Placeholder to match 'linear' activation
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class MLFinanceUtils:
    @staticmethod
    def get_lagged_features(df, n_steps, n_steps_ahead=1):
        """
        Prepares time series data for RNNs by creating lagged features.
        """
        lag_list = []
        for lag in range(n_steps + n_steps_ahead - 1, n_steps_ahead - 1, -1):
            lag_list.append(df.shift(lag))
        
        lag_array = np.dstack([i[n_steps+n_steps_ahead-1:] for i in lag_list])
        lag_array = np.swapaxes(lag_array, 1, -1)
        return lag_array

    @staticmethod
    def train_pytorch_model(model, X_train, y_train, epochs=20, lr=0.001, batch_size=32):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        
        dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        return model

def render_rnn_component():
    st.subheader("Time Series Prediction (PyTorch RNN/LSTM/GRU)")
    if st.session_state.current_data is None:
        st.warning("Please upload data first!")
        return

    df = st.session_state.current_data.select_dtypes(include=[np.number])
    if df.empty:
        st.error("No numeric columns found for time series analysis.")
        return

    target_col = st.selectbox("Select Target Column", df.columns)
    n_lags = st.slider("Number of Lags (Lookback)", 1, 30, 10)
    model_type = st.selectbox("Select RNN Type", ["LSTM", "GRU", "SimpleRNN"])
    
    if st.button("Train RNN Model"):
        with st.spinner(f"Training PyTorch {model_type}..."):
            data = df[[target_col]].values
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            X = MLFinanceUtils.get_lagged_features(pd.DataFrame(scaled_data), n_lags)
            y = scaled_data[n_lags:]
            
            # Simple train-test split
            split = int(0.8 * len(X))
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]
            
            model = PyTorchRNN(model_type, n_units=50, input_dim=X_train.shape[2])
            MLFinanceUtils.train_pytorch_model(model, X_train, y_train, epochs=10)
            
            model.eval()
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32)
                y_test_t = torch.tensor(y_test, dtype=torch.float32)
                predictions_t = model(X_test_t)
                criterion = nn.MSELoss()
                mse = criterion(predictions_t, y_test_t).item()
                
            st.success(f"Model trained! Test MSE: {mse:.6f}")
            
            predictions = predictions_t.numpy()
            pred_rescaled = scaler.inverse_transform(predictions)
            y_test_rescaled = scaler.inverse_transform(y_test)
            
            res_df = pd.DataFrame({
                'Actual': y_test_rescaled.flatten(),
                'Predicted': pred_rescaled.flatten()
            })
            st.line_chart(res_df)

def render_autoencoder_component():
    st.subheader("Dimensionality Reduction (PyTorch Autoencoders)")
    if st.session_state.current_data is None:
        st.warning("Please upload data first!")
        return

    df = st.session_state.current_data.select_dtypes(include=[np.number])
    if df.empty:
        st.error("No numeric columns found.")
        return

    cols = st.multiselect("Select columns for reduction", df.columns, default=list(df.columns[:5]))
    if not cols:
        return

    encoding_dim = st.slider("Target Dimension (Latent Space)", 1, len(cols), 2)
    
    if st.button("Train Autoencoder"):
        with st.spinner("Training PyTorch Autoencoder..."):
            data = df[cols].values
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            model = PyTorchAutoencoder(input_dim=len(cols), encoding_dim=encoding_dim)
            MLFinanceUtils.train_pytorch_model(model, scaled_data, scaled_data, epochs=20, batch_size=16)
            
            model.eval()
            with torch.no_grad():
                scaled_data_t = torch.tensor(scaled_data, dtype=torch.float32)
                encoded_data_t = model.encoder(scaled_data_t)
                encoded_data = encoded_data_t.numpy()
            
            encoded_df = pd.DataFrame(encoded_data, columns=[f"Latent_{i+1}" for i in range(encoding_dim)])
            st.write("Encoded Data (Reduced Dimensions):")
            st.dataframe(encoded_df.head())
            
            if encoding_dim >= 2:
                import plotly.express as px
                fig = px.scatter(encoded_df, x="Latent_1", y="Latent_2", title="2D Latent Space Projection")
                st.plotly_chart(fig)
