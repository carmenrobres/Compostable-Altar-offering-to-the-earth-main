# Predictive Models Documentation + Study

## Predictive Modeling of Soil Dynamics Using SARIMA and LSTM

This study presents the first attempt to predict soil behavior from experimental sensor data collected during the biomaterial altar experiment.  
Two predictive approaches were developed: a **Seasonal ARIMA (SARIMA)** model and a **Long Short-Term Memory (LSTM)** neural network.  

The models were trained on the soil sensor dataset and tested for their ability to forecast **humidity, pH, electrical conductivity (EC), and nutrient concentrations (N, P, K)**.  
A **local web interface** was built to allow dynamic forecasting, where the user uploads the first week of data and selects a prediction horizon.  

The objective is to explore early predictive capacity, not to reach full accuracy.  
With only one experiment, the model’s purpose is to test feasibility and guide future scaling.

---

## Introduction

Soil dynamics depend on complex, time-dependent factors such as temperature, irrigation, microbial activity, and material decomposition.  
While the first phase of this project focused on observing how biomaterials alter soil chemistry, this second phase introduces predictive modeling as a way to anticipate soil responses to irrigation and environmental conditions.

Using sensor data from three pots (control, clementine, and wood+clementine), two predictive approaches were developed:

- **SARIMA**: a statistical model capturing seasonal and cyclic trends.  
- **LSTM**: a machine learning model capable of learning nonlinear relationships and temporal dependencies.

This work demonstrates how early experimental data can be used to build a forecasting system.  
It also includes a local prediction web tool, which takes one week of soil sensor data as input and produces short- or long-term forecasts.

---

## Methods

### Data Source

The soil sensor data came from the **CWT NPK-PH-CTH-S** probe connected via **RS485** to a **Raspberry Pi**.  
Measurements included:

- Soil temperature, moisture, EC, pH, N, P, and K  
- Logged every six hours at 03:00, 09:00, 15:00, and 21:00  
- From three pots: *Soil (control)*, *Soil2 (clementine)*, and *Soil3 (wood+clementine)*

Before model training, days with rain over 10 mm and the irrigation-fault period were excluded to maintain consistency.

---

### Predictive System

A **local forecasting web application** was created in Python using Flask.  
The workflow is as follows:

1. The user uploads the first week of data of the soil they want predictions for.  
2. The user selects a prediction window (7, 14, or 58 days).  
3. The backend loads trained models and produces a CSV and plot of predicted values for different material recipes and the control (no biomaterial).

This setup allows **on-demand local forecasting** without internet dependency.

---

## SARIMA Model

The **Seasonal ARIMA (SARIMA)** model was chosen as a statistical baseline because it handles seasonal and cyclical patterns — ideal for soil data that repeats irrigation and daily temperature cycles.

- Each variable (**humidity, pH, EC, N, P, K**) and soil type (**soil, soil2, soil3**) was modeled separately.  
- The model used parameters **(1,1,1)(1,1,1,4)** to capture the four daily measurements (6-hour cycles).  
- **Exogenous variables** added:
  - `day_sin`, `day_cos` – daily rhythm  
  - `irrig_sin`, `irrig_cos` – 72-hour irrigation rhythm  
  - `day_norm` – long-term normalization  
  - `hour` – time of day

The SARIMA model learns from these patterns and predicts future points.  

**Error scores:**  
MAE range ≈ 0.1–0.3; RMSE < 0.4 for most variables — meaning it correctly tracks periodic cycles.

---

## LSTM Model

The **LSTM (Long Short-Term Memory)** network was introduced to model nonlinear dependencies between variables — for instance, how moisture affects EC and nutrient mobility over time.  
LSTMs are well-suited for time series because they “remember” long sequences.

**LSTM architecture:**

- Three stacked layers (128, 64, and 32 units) with dropout to avoid overfitting  
- Sequence length: 16 steps (~four days)  
- Inputs: soil sensor values, time features (hour, day), and weather context  
- Outputs: predicted humidity, pH, EC, N, P, and K for each soil type  

Each soil (soil2 and soil3) had its own trained model; the control soil was used for baseline comparison.

**Performance metrics:**

- Soil2 MAE ≈ 0.3–8.9 (depending on variable)  
- Soil3 MAE ≈ 0.1–14.3 (depending on variable)

These are expected variations given the small dataset and high dynamic range of EC and nutrients.

---

## Results

### Forecast Performance

Both models reproduced the main cycles in the data:

- Moisture and EC oscillations aligned with irrigation cycles.  
- pH and nutrient variations followed slower trends linked to biomaterial release.

SARIMA captured **regular periodic patterns** better, while LSTM captured **nonlinear fluctuations** after irrigation or drying.

---

### Comparative Behavior

- SARIMA performed better for **stable variables** like moisture and pH.  
- LSTM was more responsive to **sudden nutrient changes**, especially after irrigation spikes.  
- Combining both could provide **short-term regular forecasts (SARIMA)** and **long-term adaptive forecasts (LSTM)**.

---

### Forecast Visualization

Both systems generated daily prediction plots, with shaded areas marking the **first week of input** and the **following predicted period (7, 14, or 58 days)**.

- The **LSTM graph** (`forecast_with_input.png`) shows continuous predictions for all variables and soils.  
- The **SARIMA overlay** (`sarima_forecast_overlay.png`) confirms alignment between observed data and projected trends.

---

## Why SARIMA and LSTM Instead of Random Forest or Linear Regression

Models like **Random Forest** or **Linear Regression** are useful when data points are independent.  
However, soil sensor data are **time series**, where every value depends on what happened before (irrigation, drying, temperature changes).

- **Linear Regression** assumes linearity — soil processes are nonlinear.  
  Nutrient release, evaporation, and microbial activity change dynamically.  
- **Random Forest** captures nonlinearity but treats each record independently — it cannot “remember” past states.  
- **SARIMA** and **LSTM** are **temporal models**:
  - SARIMA captures repeating cycles like day–night or irrigation intervals.  
  - LSTM captures complex, nonlinear patterns that evolve over several days.

**Summary:**  
SARIMA explains **repeating rhythms**.  
LSTM learns **cause-and-effect relationships** over time.  
Together they model soil dynamics better than static methods.

---

## Discussion

This first predictive model shows that **soil nutrients and conductivity can be forecast** from short data windows.  
The goal was **proof of concept**, not perfect accuracy.

**Main findings:**

- Irrigation controls short-term variations in moisture and EC.  
- Biomaterials cause long-term shifts in EC and pH via nutrient release.  
- SARIMA captures regular irrigation cycles.  
- LSTM captures nonlinear responses and nutrient pulses.

The **local web interface** allows testing the models directly on a computer, without sending data to the cloud.  
This makes experimentation private and adjustable.

---

## Limitations and Future Work

This predictive system is **exploratory**, based on a single experiment. The error range of these models is very low because we are constantly assuming that irrigation happens every 72 hours. *For better predictions we would need to train the models without a fixed irrigation cycle, and make the model understand when irrigation happens and continue with the pattern.* The model should be able to work with different types of soils and different irrigations. RIght now the model is good because it is based in a very specific set of data.
To improve reliability and real-world usability:

- Repeat under varied weather and soil conditions to expand training data.  
- Calibrate sensors with laboratory soil tests for accuracy.  
- Develop a **hybrid SARIMA–LSTM model** combining periodicity and adaptability.  
- Extend to field plots with drainage to study nutrient leaching and microbial activity.

---

### Document Outline

1. Predictive Modeling of Soil Dynamics Using SARIMA and LSTM  
2. Introduction  
3. Methods  
   - Data Source  
   - Predictive System  
4. SARIMA Model  
5. LSTM Model  
6. Results  
   - Forecast Performance  
   - Comparative Behavior  
   - Forecast Visualization  
7. Why SARIMA and LSTM Instead of Random Forest or Linear Regression  
8. Discussion  
9. Limitations and Future Work
