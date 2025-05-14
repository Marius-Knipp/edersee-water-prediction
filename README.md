# Edersee Reservoir Water Level Forecast Dashboard

![Streamlit App Screenshot](https://github.com/Marius-Knipp/edersee-water-prediction/blob/main/output/images/Dashboard_screenshot.png)
*(Optional: Add a screenshot of your dashboard here. You can upload it to your GitHub repo and link it, or use an image hosting service.)*

## Project Overview

This project presents a machine learning-driven application to forecast the water levels of the Edersee reservoir in Germany one month ahead. The core of the project is an LSTM (Long Short-Term Memory) model trained on historical water level data. The forecasts are displayed through an interactive Streamlit dashboard, which is containerized using Docker and deployed on an AWS EC2 instance.

This project was undertaken as a self-learning initiative to gain practical experience with key data science tools, methodologies, and MLOps practices, including data acquisition, preprocessing, time series forecasting, model deployment, cloud infrastructure management, and CI/CD principles.

**Live Demo (if applicable):** `http://<Your-EC2-Public-IP-or-Domain>`
*(Note: If you don't intend to keep the EC2 instance running 24/7, either remove this link or add a note about its availability.)*

## Features

*   **Time Series Forecasting:** Utilizes an LSTM neural network to predict daily water levels 30 days into the future.
*   **Automated Data Pipeline:**
    *   Fetches the latest daily water level data from the official PEGELONLINE WSV API.
    *   Loads historical data from and persists updated datasets to AWS S3.
*   **Interactive Dashboard:**
    *   Built with Streamlit for a user-friendly interface.
    *   Visualizes historical water levels and future forecasts.
    *   Displays key metrics: current water level, and predicted changes over 7 and 30 days.
    *   Allows users to select a time range for viewing historical data.
*   **Cloud Deployment:**
    *   Containerized application using Docker for portability and reproducibility.
    *   Deployed on an AWS EC2 instance, demonstrating cloud infrastructure management.
    *   Utilizes AWS S3 for persistent data storage and AWS ECR for Docker image hosting.
    *   Secure access to AWS resources managed via IAM Roles.

## Data Science & Technical Highlights

This project showcases a range of data science and engineering skills:

*   **Data Acquisition & Processing:**
    *   Automated data fetching from a public API (`requests`).
    *   Robust data pipeline integrating AWS S3 for historical data storage and updates (`boto3`).
    *   Data cleaning, resampling to daily frequency, and interpolation of missing values (`pandas`).
*   **Feature Engineering:**
    *   Creation of cyclical calendar features (sine/cosine transformations for day-of-year and day-of-week) to capture seasonality (`numpy`, `pandas`).
*   **Time Series Modeling (LSTM):**
    *   Implementation of an LSTM model for sequence prediction (`tensorflow.keras`).
    *   Data scaling using `StandardScaler` (`scikit-learn`).
    *   Preparation of lookback windows for model input.
    *   Training and saving the model and scaler artifacts.
*   **Model Serving & Prediction:**
    *   Loading pre-trained model and scaler for inference.
    *   Generating 30-day ahead forecasts based on the latest available data.
*   **Data Visualization:**
    *   Interactive time series plots using `plotly` for historical data and forecasts.
    *   Custom-styled metric cards in Streamlit for key performance indicators.
*   **Software Engineering & MLOps Practices:**
    *   **Modular Code Structure:** Project organized into `src` (data, features, models), `app`, `data`, and `models` directories for maintainability.
    *   **Containerization (Docker):** `Dockerfile` to package the application and its dependencies, ensuring consistent execution across environments.
    *   **Cloud Deployment (AWS):**
        *   **EC2:** Provisioning and configuring a virtual server for application hosting.
        *   **ECR:** Storing and managing Docker images.
        *   **S3:** Scalable and durable storage for datasets.
        *   **IAM Roles:** Securely managing AWS resource access for EC2, following best practices (no hardcoded credentials in code or containers).
    *   **Dependency Management:** `requirements.txt` for Python package management.
    *   **Logging:** Implemented application-level logging for monitoring and debugging.
    *   **Version Control (Git & GitHub):** For code management and collaboration.