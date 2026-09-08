# Classical Baselines vs Deep Learning

This document records the evaluation of classical and naive statistical baselines across the 13,580 testing sequences of the dataset to anchor the performance of the deep learning architectures.

## Baseline Results

| Model | MAE (°C) | Notes |
| :--- | :--- | :--- |
| **Naive Mean** | 2.294 | Predicts the mean of the 30-day historical window. |
| **ARIMA (1,1,1)** | 1.918 | AutoRegressive Integrated Moving Average. |
| **Amazon Chronos-T5** | 1.914 | 46M Parameter Time Series Foundation Model (Zero-Shot). |
| **Naive Last** | 1.877 | Simply repeats the last observed temperature for the next 14 days. |
| **PatchTST Gaussian** | 1.691 | Custom parameterized Gaussian Head (Capacity Matched to CVAE). |
| **PatchTST-CVAE** | **1.651** | **Our Custom Probabilistic Architecture.** |
| **PatchTST (Base)** | **1.638** | **Our Custom Deterministic Architecture.** |

## Key Insights
1. **Weather is persistent:** Over a 14-day horizon, assuming "tomorrow will be exactly like today" (Naive Last) actually outperforms complex statistical models (ARIMA) and zero-shot LLMs (Amazon Chronos).
2. **Foundation Models fall short:** The massive Amazon Chronos model barely managed to match ARIMA and was actively beaten by a naive guess.
3. **Custom Deep Learning is King:** The custom-trained PatchTST and CVAE models absolutely crushed every single baseline by a massive margin (~0.22 °C tighter MAE than the best classical baseline).
