# Time-Series Data Reconstruction with Masking Strategies

## 1. Overview
This project implements various **autoencoder-based models (LSTM, GRU, and Bidirectional LSTM)** to handle **missing values in time-series data**. 
The project explores masking strategies to simulate missing data, reconstruct it, and evaluate the reconstruction accuracy using **MAE (Mean Absolute Error)**.

---

## 2. Requirements

Make sure you have **Python 3.x** installed. Install the necessary libraries using:


```pip install numpy pandas tensorflow scikit-learn matplotlib```

## 3. Python Version
Python 3.x (Tested on Python 3.8)
Uses TensorFlow 2.x.

## 4. Project Structure
```
│── 113522124.py              # Main Python script with the model and masking strategies.
│── data/                     # Folder containing sample time-series datasets.
│── README.md 
```

## 5. How to Run the Code
Prepare Data:

Place your time-series data (e.g., stock prices, sensor data) in the data/ folder.
Data should be in CSV format with relevant features (like open, close).
Run the Script: Execute the following command:

```python 113522124.py```

Output:
The code trains LSTM, GRU, and Bidirectional LSTM models.
MAE for masked values is reported to evaluate the reconstruction.

## 6. Masking Strategy in the Code
The script implements random masking to simulate missing data:

Random Masking:
Randomly masks individual time steps in the sequence, setting them to NaN.
This approach ensures that the model learns to reconstruct missing values scattered across the dataset.

## 7. Model Architectures
The script provides three models:

LSTM Autoencoder: Captures long-term dependencies.
GRU Autoencoder: Faster convergence and fewer parameters.
Bidirectional LSTM Autoencoder: Leverages both past and future context.
Each model includes an encoder-decoder architecture, where the encoder compresses the input, and the decoder reconstructs it.

## 8. Evaluation Metrics
The model's performance is evaluated using MAE (Mean Absolute Error):

Masked MAE: Only considers masked (missing) values to measure reconstruction quality.
## 9. How to Customize the Code
Change Sequence Length:
Adjust the sequence length here:

```
sequence_length = 50  # Adjust as needed
Adjust Batch Size:
Modify the batch size to balance memory usage and training speed.
```

## 10. Results and Insights
Our Masking MAE: ~0.03 to 0.04
Performance: GRU models converge faster, while Bidirectional LSTM models achieve the best reconstruction quality.
## 11. Conclusion and Recommendations
Use LSTM for long-term dependencies.
Use GRU for faster training and real-time predictions.
Use Bidirectional LSTM for tasks where both past and future context are important.
## 12. Troubleshooting
Memory Issues:

Reduce batch size or use GRU models instead of LSTM.
Overfitting:

Add dropout layers or early stopping to prevent overfitting.
