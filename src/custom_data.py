import pandas as pd


class CustomData:

    def __init__(
        self,
        gender,
        SeniorCitizen,
        Partner,
        Dependents,
        tenure,
        PhoneService,
        MultipleLines,
        InternetService,
        OnlineSecurity,
        OnlineBackup,
        DeviceProtection,
        TechSupport,
        StreamingTV,
        StreamingMovies,
        Contract,
        PaperlessBilling,
        PaymentMethod,
        MonthlyCharges,
        TotalCharges
    ):

        self.gender = gender
        self.SeniorCitizen = SeniorCitizen
        self.Partner = Partner
        self.Dependents = Dependents
        self.tenure = tenure
        self.PhoneService = PhoneService
        self.MultipleLines = MultipleLines
        self.InternetService = InternetService
        self.OnlineSecurity = OnlineSecurity
        self.OnlineBackup = OnlineBackup
        self.DeviceProtection = DeviceProtection
        self.TechSupport = TechSupport
        self.StreamingTV = StreamingTV
        self.StreamingMovies = StreamingMovies
        self.Contract = Contract
        self.PaperlessBilling = PaperlessBilling
        self.PaymentMethod = PaymentMethod
        self.MonthlyCharges = MonthlyCharges
        self.TotalCharges = TotalCharges

    def get_dataframe(self):

        data = {

            "gender":[self.gender],
            "SeniorCitizen":[self.SeniorCitizen],
            "Partner":[self.Partner],
            "Dependents":[self.Dependents],
            "tenure":[self.tenure],
            "PhoneService":[self.PhoneService],
            "MultipleLines":[self.MultipleLines],
            "InternetService":[self.InternetService],
            "OnlineSecurity":[self.OnlineSecurity],
            "OnlineBackup":[self.OnlineBackup],
            "DeviceProtection":[self.DeviceProtection],
            "TechSupport":[self.TechSupport],
            "StreamingTV":[self.StreamingTV],
            "StreamingMovies":[self.StreamingMovies],
            "Contract":[self.Contract],
            "PaperlessBilling":[self.PaperlessBilling],
            "PaymentMethod":[self.PaymentMethod],
            "MonthlyCharges":[self.MonthlyCharges],
            "TotalCharges":[self.TotalCharges]

        }

        return pd.DataFrame(data)