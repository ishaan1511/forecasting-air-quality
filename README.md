# Forecasting Air Quality

## 1. Inspiration

Air quality has a direct and measurable impact on public health, yet many existing tools focus primarily on reporting historical conditions rather than anticipating future trends. This project explores whether a compact, data-driven neural network can reliably forecast near-term PM2.5 concentrations using historical observations and time-based signals. Rather than attempting to model atmospheric physics explicitly, the goal is to learn how air pollution evolves over time from data alone, producing practical short-horizon forecasts that could support awareness and decision-making.

---

## 2. Working with Real Air Quality Data

This project uses publicly available hourly air quality data containing PM2.5 measurements alongside temporal information. As is common with environmental datasets, the raw data included missing values — approximately 2,000 PM2.5 readings across 43,000 hourly time steps. Because sequence models require continuous inputs, missing values were handled using forward and backward filling to preserve temporal continuity while minimizing distortion of underlying trends. To capture periodic pollution patterns, we engineered cyclical time features by applying sine and cosine transformations to hour, day, and month variables. These features allow the model to learn daily and seasonal dynamics in air quality. After cleaning and feature construction, the dataset was reshaped into fixed-length sequences suitable for recurrent neural networks. This preprocessing step was critical for enabling the model to learn meaningful temporal dependencies and produce stable forecasts.

---
## 3. Modeling Approach

To establish a fair and informative comparison, we evaluate three commonly used recurrent architectures for time-series forecasting: a simple Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). Each model is trained and evaluated under identical data preprocessing, sequence length, and training conditions to ensure comparability.

Model performance is assessed using Root Mean Squared Error (RMSE) on a held-out validation set, as RMSE directly reflects the magnitude of prediction error in PM2.5 concentration. The architecture achieving the lowest validation RMSE is selected as the base model for further experimentation.

Once the best-performing recurrent architecture is identified, we fine-tune it through targeted adjustments to hyperparameters such as hidden size, number of layers, learning rate, and regularization. This two-stage approach—broad model comparison followed by focused fine-tuning—ensures that performance gains are driven by architectural suitability rather than differences in training conditions.


## 6. References

- **Beijing PM2.5 Dataset (UCI Machine Learning Repository)**  
  Primary data source containing hourly PM2.5 measurements and meteorological features.  
  [https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data](https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data)
