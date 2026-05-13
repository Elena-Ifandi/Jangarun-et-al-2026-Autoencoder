# Import required libraries
import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.model_selection import train_test_split, KFold
import torch
from torch import nn
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

def evaluate_model(model, data_loader):
    """Evaluate model performance and return metrics"""
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data in data_loader:
            inputs = data[0]
            reconstructed = model(inputs)
            all_preds.append(reconstructed)
            all_targets.append(inputs)

    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    mse = mean_squared_error(all_targets, all_preds)
    r2 = r2_score(all_targets, all_preds)
    return {'mse': mse, 'r2': r2}

# Step 1: Preprocessing Pipeline
def preprocess_data():
    print("Preprocessing data...")
    df = pd.read_csv('Dataset.csv')

    # Split data first to prevent data leakage
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Calculate percentile ranks separately for train and test
    def process_split(split_df):
        original_values = split_df.copy()
        dropped_columns = split_df[['Level', 'Data']]
        processed_df = split_df.drop(['Level', 'Data'], axis=1)
        processed_df = processed_df.apply(
            lambda x: rankdata(x, method='average') / len(x)
            if x.dtype != 'object' else x
        )
        processed_df = pd.concat([processed_df, dropped_columns], axis=1)
        return processed_df, original_values

    train_processed, train_original = process_split(train_df)
    test_processed, test_original = process_split(test_df)

    # Split by soil type
    def split_by_soil(processed_df, original_df):
        return {
            'Topsoil': (
                processed_df[processed_df['Level'] == 'Topsoil'].drop(['Level', 'Data'], axis=1),
                original_df[original_df['Level'] == 'Topsoil'].drop(['Level', 'Data'], axis=1)
            ),
            'Subsoil': (
                processed_df[processed_df['Level'] == 'Subsoil'].drop(['Level', 'Data'], axis=1),
                original_df[original_df['Level'] == 'Subsoil'].drop(['Level', 'Data'], axis=1)
            )
        }

    train_soil_dfs = split_by_soil(train_processed, train_original)
    test_soil_dfs = split_by_soil(test_processed, test_original)

    return train_soil_dfs, test_soil_dfs

# Step 2: Updated Model Architecture
class ImprovedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(ImprovedAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    @staticmethod
    def train_epoch(model, train_loader, val_loader, criterion, optimizer, scheduler, epoch):
        """Train the model for one epoch and evaluate on validation data"""
        # Training phase
        model.train()
        running_train_loss = 0.0
    
        for data in train_loader:
            # Get the inputs
            inputs = data[0]
        
            # Zero the parameter gradients
            optimizer.zero_grad()
        
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
        
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
        
            running_train_loss += loss.item() * inputs.size(0)
    
        # Calculate average training loss for this epoch
        train_loss = running_train_loss / len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():
            for data in val_loader:
                inputs = data[0]
                outputs = model(inputs)
                val_loss = criterion(outputs, inputs)
                running_val_loss += val_loss.item() * inputs.size(0)
        
        # Calculate average validation loss for this epoch
        val_loss = running_val_loss / len(val_loader.dataset)
        
        # Step the learning rate scheduler based on validation loss
        if scheduler:
            scheduler.step(val_loss)
        
        return train_loss, val_loss


    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Step 3: Hybrid Loss Function
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.mse(y_pred, y_true) + (1 - self.alpha) * self.l1(y_pred, y_true)

# Step 4: Mini-batch Training with Learning Rate Scheduling
def train_model(model, train_loader, val_loader, epochs=200):
    print("Training autoencoder...")
    criterion = HybridLoss(alpha=0.7)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Use the train_epoch method from ImprovedAutoencoder
        train_loss, val_loss = ImprovedAutoencoder.train_epoch(
            model, train_loader, val_loader, criterion, optimizer, scheduler, epoch
        )
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Learning rate scheduling is handled within train_epoch
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Final evaluation
    train_metrics = evaluate_model(model, train_loader)
    val_metrics = evaluate_model(model, val_loader)
    
    return model, train_metrics, val_metrics

    
def train_model_cv(data, batch_size=32, n_splits=10, num_epochs=200, patience=20):
    print("Training autoencoder with 10-fold cross-validation...")
    input_dim = data.shape[1]
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        print(f"--- Fold {fold+1}/{n_splits} ---")
        train_data, val_data = data.iloc[train_index], data.iloc[val_index]

        model = ImprovedAutoencoder(input_dim)
        criterion = HybridLoss(alpha=0.7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        train_tensor = torch.tensor(train_data.values, dtype=torch.float32)
        val_tensor = torch.tensor(val_data.values, dtype=torch.float32)
        train_dataset = TensorDataset(train_tensor)
        val_dataset = TensorDataset(val_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(num_epochs):
             train_loss, val_loss = ImprovedAutoencoder.train_epoch(
    model, train_loader, val_loader, criterion, optimizer, scheduler, epoch
)


             if epoch % 50 == 0: # Print less often for CV folds
                print(f"Epoch {epoch}: Train loss = {train_loss:.4f}, Val loss = {val_loss:.4f}")

             # Early stopping
             if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
             else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch} for fold {fold+1}")
                    break

        # Evaluate on the validation set of the current fold
        fold_metrics = evaluate_model(model, val_loader)
        fold_results.append(fold_metrics)
        print(f"Fold {fold+1} Validation Metrics: {fold_metrics}")

    # Calculate average metrics across all folds
    avg_metrics = {metric: np.mean([res[metric] for res in fold_results]) for metric in fold_results[0].keys()}
    print("\nAverage Cross-Validation Metrics:", avg_metrics)

    return avg_metrics


def calculate_stats(model, data):
    print("Calculating stats...")
    n_bootstrap = 1000
    bootstrapped_data = []

    for _ in range(n_bootstrap):
        indices = np.random.choice(len(data), size=len(data), replace=True)
        bootstrap_sample = data.iloc[indices]
        data_tensor = torch.tensor(bootstrap_sample.values, dtype=torch.float32)

        with torch.no_grad():
            reconstructed = model(data_tensor).numpy()
            bootstrapped_data.append(reconstructed)

    bootstrapped_array = np.concatenate(bootstrapped_data)
    stats = {}

    for col in range(data.shape[1]):
        values = bootstrapped_array[:, col]
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        stats[data.columns[col]] = {
            'Q1': q1,
            'Q3': q3,
            'IQR': iqr
        }

    return stats
    
def convert_stats_to_ppm(stats, original_data):
    ppm_stats = {}
    for column in original_data.columns:
        original_values = original_data[column].values

        # Convert percentile ranks to PPM values
        sorted_vals = np.sort(original_values)
        q1_idx = int(stats[column]['Q1'] * (len(sorted_vals) - 1))
        q3_idx = int(stats[column]['Q3'] * (len(sorted_vals) - 1))

        q1_ppm = sorted_vals[q1_idx]
        q3_ppm = sorted_vals[q3_idx]
        iqr_ppm = q3_ppm - q1_ppm

        ppm_stats[column] = {
            'Q1': q1_ppm,
            'Q3': q3_ppm,
            'IQR': iqr_ppm
        }
    return ppm_stats
    
def calculate_ppm_thresholds(ppm_stats):
    ppm_thresholds = {}
    for column, stats in ppm_stats.items():
        q1 = stats['Q1']
        q3 = stats['Q3']
        iqr = stats['IQR']

        ppm_thresholds[column] = {
            'outer_upper': q3 + (3.0 * iqr),
            'inner_upper': q3 + (1.5 * iqr),
            'inner_lower': max(q1 - 1.5 * iqr, 0),
            'outer_lower': max(q1 - 3.0 * iqr, 0)
        }
    return ppm_thresholds

    

def train_model_single_split(train_df, test_df, batch_size=32):
    # Create tensors
    train_tensor = torch.tensor(train_df.values, dtype=torch.float32)
    test_tensor = torch.tensor(test_df.values, dtype=torch.float32)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = train_df.shape[1]
    model = ImprovedAutoencoder(input_dim)
    
    # Train model
    model, train_metrics, val_metrics = train_model(model, train_loader, test_loader)
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader)
    
    return model, train_metrics, test_metrics



    
def create_plots(soil_type, original_data, ppm_thresholds):
    print(f"Creating plots for {soil_type}...")
    colors = {
        'inner_lower': 'orange',
        'inner_upper': 'orange',
        'outer_lower': 'red',
        'outer_upper': 'red'
    }

    for column in original_data.columns:
        plt.figure(figsize=(12, 6))

        thresholds = ppm_thresholds[column]

        outer_mask = (original_data[column] < thresholds['outer_lower']) | (original_data[column] > thresholds['outer_upper'])
        inner_mask = (original_data[column] < thresholds['inner_lower']) | (original_data[column] > thresholds['inner_upper'])
        normal_mask = ~inner_mask & ~outer_mask

        plt.scatter(np.where(outer_mask)[0], original_data[column][outer_mask],
                   color='red', alpha=0.5, label='Extreme Outliers')
        plt.scatter(np.where(inner_mask & ~outer_mask)[0], original_data[column][inner_mask & ~outer_mask],
                   color='orange', alpha=0.5, label='Mild Outliers')
        plt.scatter(np.where(normal_mask)[0], original_data[column][normal_mask],
                   color='green', alpha=0.5, label='Normal Range')

        for threshold, value in thresholds.items():
            plt.axhline(y=value, linestyle='--', color=colors[threshold],
                       label=f'{threshold} threshold')

        plt.title(f'{soil_type} - {column} Concentration')
        plt.xlabel('Sample Index')
        plt.ylabel('Concentration (PPM)')
        plt.legend()
        plt.savefig(f'{soil_type}_{column}_thresholds.png')
        plt.close()
    
    for column in original_data.columns:
        fig = go.Figure()

        thresholds = ppm_thresholds[column]

        outer_mask = (original_data[column] < thresholds['outer_lower']) | (original_data[column] > thresholds['outer_upper'])
        inner_mask = (original_data[column] < thresholds['inner_lower']) | (original_data[column] > thresholds['inner_upper'])
        normal_mask = ~inner_mask & ~outer_mask

        fig.add_trace(go.Scatter(
            x=np.where(outer_mask)[0],
            y=original_data[column][outer_mask],
            mode='markers',
            name='Extreme Outliers',
            marker=dict(color='red', opacity=0.5)
        ))

        fig.add_trace(go.Scatter(
            x=np.where(inner_mask & ~outer_mask)[0],
            y=original_data[column][inner_mask & ~outer_mask],
            mode='markers',
            name='Mild Outliers',
            marker=dict(color='orange', opacity=0.5)
        ))

        fig.add_trace(go.Scatter(
            x=np.where(normal_mask)[0],
            y=original_data[column][normal_mask],
            mode='markers',
            name='Normal Range',
            marker=dict(color='green', opacity=0.5)
        ))

        for threshold, value in thresholds.items():
            fig.add_hline(
                y=value,
                line_dash="dash",
                line_color=colors[threshold],
                annotation_text=f"{threshold} threshold"
            )

        fig.update_layout(
            title=f'{soil_type} - {column} Concentration',
            xaxis_title='Sample Index',
            yaxis_title='Concentration (PPM)',
            showlegend=True,
            width=1200,
            height=960
        )

        fig.write_html(f'{soil_type}_{column}_thresholds.html')
        fig.write_image(f'{soil_type}_{column}_thresholds.png')
        plt.close()

def main():
    # Execute pipeline
    train_soil_dfs, test_soil_dfs = preprocess_data()
    #print(train_soil_dfs)

    results = {}
    for soil_type in train_soil_dfs.keys():
        print(f"\nProcessing {soil_type}...")

        train_df, train_original = train_soil_dfs[soil_type]
        test_df, test_original = test_soil_dfs[soil_type]

        print(f"\nProcessing {soil_type}... train_df")

        print(train_df)

        # Train model with 10-fold cross-validation
        avg_cv_metrics = train_model_cv(train_df)

        # Train final model on full training data
        print(f"\nTraining {soil_type} final model on the full training data for stats calculation...")
        final_model_for_stats, train_metrics, test_metrics = train_model_single_split(train_df, test_df)


        # Calculate stats using the final model trained on the full training data
        stats = calculate_stats(final_model_for_stats, train_df)

        # Convert stats to PPM
        ppm_stats = convert_stats_to_ppm(stats, train_original)

        # Calculate thresholds using converted PPM stats
        ppm_thresholds = calculate_ppm_thresholds(ppm_stats)

        # Store results
        results[soil_type] = {
            'avg_cv_metrics': avg_cv_metrics,
            'train_metrics': train_metrics,
            'test_metrics': test_metrics,
            #'bootstrap_metrics': bootstrap_metrics,
            'thresholds': ppm_thresholds,
            'stats': ppm_stats
        }

        # Export to CSV
        combined_results = {}
        for column in train_original.columns:
            combined_results[column] = {
                **ppm_thresholds[column],
                **ppm_stats[column]
            }

        pd.DataFrame(combined_results).to_csv(f'{soil_type}_thresholds_and_stats.csv')

        print(f"\n{soil_type} Results:")
        print("Average Cross-Validation Metrics:", avg_cv_metrics)
        print("Train Metrics:", train_metrics)
        print("Test Metrics:", test_metrics)
        print("\nThresholds (PPM):")
        for param, values in ppm_thresholds.items():
            print(f"\n{param}:")
            for threshold, value in values.items():
                print(f"{threshold}: {value:.2f}")
                
         
        # Create plots
        create_plots(soil_type, test_original, ppm_thresholds)

if __name__ == "__main__":
    main()




            
            
