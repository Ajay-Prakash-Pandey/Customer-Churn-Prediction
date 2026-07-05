from src.custom_data import CustomData

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

print(df)

print("\nShape")

print(df.shape)