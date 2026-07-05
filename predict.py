from src.custom_data import CustomData
from src.components.prediction_pipeline import PredictionPipeline


customer = CustomData(

    gender="Male",
    SeniorCitizen=0,
    Partner="Yes",
    Dependents="No",
    tenure=24,
    PhoneService="Yes",
    MultipleLines="Yes",
    InternetService="Fiber optic",
    OnlineSecurity="No",
    OnlineBackup="Yes",
    DeviceProtection="Yes",
    TechSupport="No",
    StreamingTV="Yes",
    StreamingMovies="Yes",
    Contract="Month-to-month",
    PaperlessBilling="Yes",
    PaymentMethod="Electronic check",
    MonthlyCharges=85.5,
    TotalCharges=2052.0

)

df = customer.get_dataframe()

pipeline = PredictionPipeline()

prediction, probability = pipeline.predict(df)

print("=" * 50)

print("Prediction")

print("=" * 50)

prediction_value = int(prediction[0])

print("Prediction :", "Churn" if prediction_value == 1 else "No Churn")

if probability is not None:
    print(f"Probability : {probability[0][1]:.2%}")
