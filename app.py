import streamlit as st
import joblib
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
model = joblib.load("data_model.pkl")


df = pd.read_csv("C:\\Users\\sharm\\OneDrive\\Desktop\\CustomerChurn-selected-columns.csv")

# Data Cleaning
df["Total Charges"] = df["Total Charges"].replace(" ", pd.NA)
df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
df.dropna(inplace=True)
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction System")

st.write("""
Welcome to the Customer Churn Prediction System.

This application predicts whether a customer is likely to churn based on different customer attributes.

Use the navigation menu on the left to explore the application.
""")

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a Page",
    [
        "🏠 Home",
        "🤖 Customer Prediction",
        "📈 Graphical Analysis",
        "ℹ Overview"
    ]
)

if page == "🏠 Home":
    st.header("📌 Home")
    st.write("Welcome to the Customer Churn Prediction System.")

elif page == "🤖 Customer Prediction":
    st.header("⚙️ Customer Churn Prediction")
    col1, col2 = st.columns(2)

    with col1:

        senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
        senior_citizen = 1 if senior_citizen == "Yes" else 0

        partner = st.selectbox("Partner", ["No", "Yes"])
        partner = 1 if partner == "Yes" else 0

        dependents = st.selectbox("Dependents", ["No", "Yes"])
        dependents = 1 if dependents == "Yes" else 0

        tenure = st.number_input("Tenure", min_value=0, step=1)

        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )
        internet_service = {
            "DSL": 0,
            "Fiber optic": 1,
            "No": 2
        }[internet_service]

        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        contract = {
            "Month-to-month": 0,
            "One year": 1,
            "Two year": 2
        }[contract]

        monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

    with col2:

        online_security = st.selectbox("Online Security", ["No", "Yes"])
        online_security = 1 if online_security == "Yes" else 0

        online_backup = st.selectbox("Online Backup", ["No", "Yes"])
        online_backup = 1 if online_backup == "Yes" else 0

        device_protection = st.selectbox("Device Protection", ["No", "Yes"])
        device_protection = 1 if device_protection == "Yes" else 0

        tech_support = st.selectbox("Tech Support", ["No", "Yes"])
        tech_support = 1 if tech_support == "Yes" else 0

        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"])
        streaming_tv = 1 if streaming_tv == "Yes" else 0

        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"])
        streaming_movies = 1 if streaming_movies == "Yes" else 0

        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
        paperless_billing = 1 if paperless_billing == "Yes" else 0

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        payment_method = {
            "Electronic check": 2,
            "Mailed check": 3,
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 1
        }[payment_method]
        total_charges = st.number_input("Total Charges", min_value=0.0)
    if st.button("🔍 Predict Churn", use_container_width=True):
        input_data= pd.DataFrame([[senior_citizen,partner,dependents,tenure,internet_service,online_security,
            online_backup,device_protection,tech_support,streaming_tv,streaming_movies,contract,paperless_billing,payment_method,
            monthly_charges,total_charges
        ]], columns=['Senior Citizen','Partner','Dependents','Tenure','Internet Service','Online Security','Online Backup',
            'Device Protection','Tech Support','Streaming TV','Streaming Movies','Contract','Paperless Billing',
            'Payment Method','Monthly Charges','Total Charges'
        ])

        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)

        churn_probability = probability[0][1] * 100
        stay_probability = probability[0][0] * 100

        if churn_probability >= 80:
          risk = "🔴 High Risk"

        elif churn_probability >= 50:
         risk = "🟠 Medium Risk"

        else:
         risk = "🟢 Low Risk"


        if prediction[0] == 1:

            st.error("⚠️ Customer is likely to Churn.")

            st.metric("Churn Probability", f"{churn_probability:.2f}%")

            st.write(f"### Churn Risk: {risk}")

            st.progress(churn_probability/100)

            st.warning("""
        ### Recommended Actions

        • Offer special discounts or loyalty benefits.

        • Improve customer support.

        • Contact the customer personally.

        • Provide attractive subscription plans.

        • Resolve service-related complaints quickly.
        """)

        else:

            st.success("✅ Customer is likely to Stay.")

            st.metric("Stay Probability", f"{stay_probability:.2f}%")

            st.progress(stay_probability/100)

            st.write(f"### Churn Risk: {risk}")

            st.info("""
        ### Recommendation

        • Continue providing good customer service.

        • Offer loyalty rewards.

        • Maintain customer satisfaction.

        • Introduce premium plans for long-term engagement.
        """)

elif page == "📈 Graphical Analysis":

    st.header("📈 Graphical Analysis")

    graph = st.selectbox(
        "Select Graph",
        (
            "Churn Distribution",
            "Contract vs Churn",
            "Monthly Charges Distribution",
            "Tenure Distribution",
            "Internet Service vs Churn",
            "Payment Method vs Churn",
            "Correlation Heatmap"
        )
    )

    if graph == "Churn Distribution":

        fig, ax = plt.subplots(figsize=(6,5))
        sns.countplot(x="Churn", data=df, ax=ax)
        ax.set_title("Customer Churn Distribution")
        st.pyplot(fig)
        st.info("This graph shows the number of customers who stayed and those who churned.")

    elif graph == "Contract vs Churn":

        fig, ax = plt.subplots(figsize=(7,5))
        sns.countplot(x="Contract", hue="Churn", data=df, ax=ax)
        ax.set_title("Contract vs Churn")
        st.pyplot(fig)
        st.info("Customers with month-to-month contracts show higher churn than customers with yearly contracts.")

    elif graph == "Monthly Charges Distribution":

        fig, ax = plt.subplots(figsize=(7,5))
        sns.histplot(df["Monthly Charges"], bins=30, kde=True, ax=ax)
        ax.set_title("Monthly Charges Distribution")
        st.pyplot(fig)
        st.info("This graph shows the distribution of monthly charges among customers.")
    elif graph == "Tenure Distribution":

        fig, ax = plt.subplots(figsize=(7,5))
        sns.histplot(df["Tenure"], bins=30, kde=True, ax=ax)
        ax.set_title("Tenure Distribution")
        st.pyplot(fig)
        st.info("This graph shows how long customers have stayed with the company.")
    elif graph == "Internet Service vs Churn":

        fig, ax = plt.subplots(figsize=(7,5))
        sns.countplot(x="Internet Service", hue="Churn", data=df, ax=ax)
        ax.set_title("Internet Service vs Churn")
        st.pyplot(fig)
        st.info("This graph compares customer churn across different internet service types.")

    elif graph == "Payment Method vs Churn":

        fig, ax = plt.subplots(figsize=(8,5))
        sns.countplot(x="Payment Method", hue="Churn", data=df, ax=ax)
        plt.xticks(rotation=20)
        ax.set_title("Payment Method vs Churn")
        st.pyplot(fig)
        st.info("This graph shows the relationship between payment methods and customer churn.")

    elif graph == "Correlation Heatmap":

        graph_df = df.copy()

        graph_df = graph_df.drop("Customer ID", axis=1)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()

        categorical_col = [
            'Senior Citizen','Partner','Dependents','Internet Service',
            'Online Security','Online Backup','Device Protection',
            'Tech Support','Streaming TV','Streaming Movies',
            'Contract','Paperless Billing','Payment Method','Churn'
        ]

        for col in categorical_col:
            graph_df[col] = le.fit_transform(graph_df[col])

        fig, ax = plt.subplots(figsize=(16,12))

        sns.heatmap(
        graph_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        annot_kws={"size":8},
        linewidths=0.5,
        ax=ax
        )

        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
 
        st.pyplot(fig)
        st.info("The heatmap displays the correlation between different features in the dataset.")

elif page == "ℹ Overview":
    st.header("Overview")
    st.write("""
    This project predicts customer churn using Machine Learning.

    Models Used:
    - Decision Tree
    - Logistic Regression
    - Random Forest

    Developed using Python, Scikit-learn and Streamlit.
    """)

if page == "🏠 Home":

    st.header("📌 Customer Churn Prediction System")

    st.markdown("""
### Welcome 👋

This web application predicts whether a customer is likely to churn based on customer information.

The system helps businesses identify customers who are likely to leave so that they can take preventive actions and improve customer retention.
""")

    col1, col2 = st.columns(2)

    with col1:
        st.info("""
### 🎯 Project Objective

Predict customer churn using machine learning and help organizations identify customers who are likely to leave.
""")

    with col2:
        st.info("""
### 🚀 Features

✅ Predict Customer Churn

✅ Interactive Graphical Analysis

✅ Correlation Heatmap

✅ User-Friendly Interface
""")

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    st.write("""
• Customer Churn Dataset

• 16 Input Features

• Target Variable: Churn

• Cleaned and Preprocessed Dataset
""")

    st.markdown("---")

    st.subheader("📝 How to Use")
    st.markdown("---")

    st.success("Thank you for using the Customer Churn Prediction System!")

    st.write("""
1. Open **Customer Prediction** from the sidebar.
2. Enter the customer details.
3. Click **Predict Churn**.
4. View the prediction result.
5. Explore graphs in the **Graphical Analysis** section.
""")
elif page == "ℹ Overview":

    st.header("ℹ Overview")

    st.subheader("📌 Problem Statement")
    st.write("""
Customer churn is one of the biggest challenges faced by telecom companies.
This project predicts whether a customer is likely to leave the company based on customer-related information.
""")

    st.subheader("🎯 Objective")
    st.write("""
The objective of this project is to help organizations identify customers who are likely to churn so that appropriate retention strategies can be implemented.
""")

    st.subheader("🤖 Machine Learning Models")
    st.write("""
• Decision Tree

• Logistic Regression

• Random Forest
""")

    st.subheader("⚙ Technologies Used")
    st.write("""
• Python

• Pandas

• NumPy

• Scikit-learn

• Streamlit

• Matplotlib

• Seaborn

• Joblib
""")

    st.subheader("🚀 Future Scope")
    st.write("""
• Improve prediction accuracy using advanced machine learning models.

• Deploy the application on the cloud.

• Integrate with real-time customer databases.

• Provide personalized customer retention suggestions.
""")
    
#python -m streamlit run app.py