# Edersee Reservoir Water Level Forecast Dashboard

![Streamlit App Screenshot](https://github.com/Marius-Knipp/edersee-water-prediction/blob/main/output/images/Dashboard_screenshot.png)

## Project Overview

This project presents a machine learning-driven application to forecast the water levels of the Edersee reservoir in Germany one month ahead. The core of the project is an LSTM (Long Short-Term Memory) model trained on historical water level data. The forecasts are displayed through an interactive Streamlit dashboard, which is containerized using Docker and deployed on an AWS EC2 instance.

This project was undertaken as a self-learning initiative to gain practical experience with key data science tools, methodologies, and MLOps practices, including data acquisition, preprocessing, time series forecasting, model deployment, cloud infrastructure management, and CI/CD principles.

**Live Demo:** [Streamlit App](http://18.197.84.254/)

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


## Challenges & Learning

*   **Debugging Cloud Deployments:** Encountered initial challenges with AWS App Runner logging and IAM role configurations, leading to a successful pivot to EC2 for more granular control and debugging access. This highlighted the importance of understanding different cloud service models (PaaS vs. IaaS).
*   **IAM Permissions:** Gained a deeper understanding of IAM roles, policies, and trust relationships for secure AWS resource access, particularly for ECR and S3 interactions from compute services.
*   **Docker Networking & Port Mapping:** Practical experience with exposing container ports and mapping them to host ports.
*   **Streamlit in Docker:** Ensured Streamlit server configurations (`--server.address=0.0.0.0`, `--server.headless=true`) were appropriate for a containerized environment.
*   **Credential Management:** Successfully implemented credential handling via IAM roles for EC2 deployment, avoiding hardcoded keys. For local Docker testing, used environment variables.

## Future Enhancements

*   **HTTPS & Custom Domain:** Implement HTTPS using AWS Certificate Manager and an Application Load Balancer, or Nginx + Let's Encrypt on the EC2 instance, and point a custom domain.
*   **CI/CD Pipeline:** Automate the build, test, and deployment process using GitHub Actions or AWS CodePipeline.
*   **Centralized Logging:** Configure Docker to send container logs to AWS CloudWatch Logs.
*   **Scheduled Data Updates:** Decouple the S3 data update process using AWS Lambda and EventBridge for a more robust pipeline.
*   **Model Retraining & Versioning:** Implement a strategy for periodic model retraining and versioning (e.g., using MLflow or Sagemaker).
*   **Advanced Health Checks:** Create a dedicated health check endpoint in the Streamlit app.
*   **Cost Optimization:** Explore more cost-effective EC2 instance types or serverless options like AWS Fargate if scaling becomes a need.
