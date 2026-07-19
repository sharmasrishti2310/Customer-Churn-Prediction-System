import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
import joblib
warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)


df=pd.read_csv("C:\\Users\\sharm\\OneDrive\\Desktop\\CustomerChurn-selected-columns.csv")
#print(df.head())
# Check blank spaces
# print("Blank Spaces:", (df['Total Charges'] == " ").sum())

# Replace blank spaces with NaN
df['Total Charges'] = df['Total Charges'].replace(" ", np.nan)

# Convert to numeric
df['Total Charges'] = pd.to_numeric(df['Total Charges'], errors='coerce')

# Remove rows having NaN
df.dropna(inplace=True)
"""print(df["Internet Service"].unique())
print(df["Contract"].unique())
print(df["Payment Method"].unique())"""

# Verify
#print("Missing Values:", df['Total Charges'].isnull().sum())
#print("Blank Spaces After Cleaning:", (df['Total Charges'].astype(str) == " ").sum())
graph_df = df.copy()
le=LabelEncoder()
categorical_col=['Senior Citizen','Partner','Dependents','Internet Service','Online Security',
                 'Online Backup','Device Protection','Tech Support','Streaming TV','Streaming Movies',
                 'Contract','Paperless Billing','Payment Method','Churn']
for col in categorical_col:
    df[col]=le.fit_transform(df[col])

X=df.drop(['Customer ID','Churn'],axis=1)
Y=df['Churn']


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
dt=DecisionTreeClassifier(random_state=42)
dt.fit(X_train,Y_train)
train_predictions=dt.predict(X_train)
test_predictions=dt.predict(X_test)
train_accuracy=accuracy_score(Y_train,train_predictions)
test_accuracy=accuracy_score(Y_test,test_predictions)
print("Training Accuracy:",train_accuracy)
print("Testing Accuracy:",test_accuracy)
print(classification_report(Y_test, test_predictions))


# Logistic Regression
"""model_lr=LogisticRegression(max_iter=1000)
model_lr.fit(X_train_scaled,Y_train)
pred_train=model_lr.predict(X_train_scaled)
pred_test=model_lr.predict(X_test_scaled)
accuracy_train=accuracy_score(Y_train,pred_train)
accuracy_test=accuracy_score(Y_test,pred_test)
Classification_report=classification_report(Y_test,pred_test)
print(Classification_report)
print(accuracy_train)
print(accuracy_test)

model_rf=RandomForestClassifier(random_state=42)
model_rf.fit(X_train,Y_train)
pre_train=model_rf.predict(X_train)
pre_test=model_rf.predict(X_test)
acc_train=accuracy_score(Y_train,pre_train)
acc_test=accuracy_score(Y_test,pre_test)
Report=classification_report(Y_test,pre_test)
print(acc_train)
print(acc_test)
print(Report)

comparison = pd.DataFrame({
    "Model": ["Decision Tree", "Logistic Regression", "Random Forest"],
    "Accuracy": [test_accuracy, accuracy_test, acc_test],
    "Precision": [
        precision_score(Y_test, test_predictions),
        precision_score(Y_test, pred_test),
        precision_score(Y_test, pre_test)
    ],
    "Recall": [
        recall_score(Y_test, test_predictions),
        recall_score(Y_test, pred_test),
        recall_score(Y_test, pre_test)
    ],
    "F1 Score": [
        f1_score(Y_test, test_predictions),
        f1_score(Y_test, pred_test),
        f1_score(Y_test, pre_test)
    ]
})

#print(comparison)

best = comparison.loc[comparison["Accuracy"].idxmax()]
#print("\nBest Model")
#print("Model :", best["Model"])
#print("Accuracy :", best["Accuracy"])

# K FOLD
scores = cross_val_score(model_rf, X, Y, cv=5)

#print("Cross Validation Scores:", scores)
#print("Average Accuracy:", scores.mean())

# Parameter Grid for Random Forest
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Create Random Forest Model
ran_model = RandomForestClassifier(random_state=42)

# Grid Search with 5-Fold Cross Validation
grid_search = GridSearchCV(
    estimator=ran_model,
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

# Fit on training data
grid_search.fit(X_train, Y_train)

# Best Parameters
#print("Best Parameters:", grid_search.best_params_)

# Best F1 Score
#print("Best F1 Score:", grid_search.best_score_)

# Best Model
best_model = grid_search.best_estimator_

joblib.dump(best_model, "data_model.pkl")
print("model successful")

# Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    best_model,
    X_train,
    Y_train,
    cv=5,
    scoring="accuracy"
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

def model_analysis():
    ...
    print("Best Model:", best["Model"])
    print("Cross Validation Accuracy:", scores.mean())
    print("Best Parameters:", grid_search.best_params_)

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_mean, marker='o', label='Training Score')
    plt.plot(train_sizes, test_mean, marker='o', label='Validation Score')
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.grid(True)
    plt.show()

def graphical_analysis():

    while True:

        print("\n========== GRAPHICAL ANALYSIS ==========")
        print("1. Churn Distribution")
        print("2. Contract vs Churn")
        print("3. Monthly Charges Distribution")
        print("4. Tenure Distribution")
        print("5. Internet Service vs Churn")
        print("6. Payment Method vs Churn")
        print("7. Correlation Heatmap")
        print("8. Back")

        choice = input("Enter your choice: ")

        if choice == "1":

            plt.figure(figsize=(6,5))
            sns.countplot(x='Churn', data=graph_df)
            plt.title("Customer Churn Distribution")
            plt.show()


        elif choice == "2":

            plt.figure(figsize=(7,5))
            sns.countplot(x='Contract', hue='Churn', data=graph_df)
            plt.title("Contract Type vs Churn")
            plt.show()


        elif choice == "3":

            plt.figure(figsize=(7,5))
            sns.histplot(df["Monthly Charges"], bins=30, kde=True)
            plt.title("Monthly Charges Distribution")
            plt.xlabel("Monthly Charges")
            plt.show()


        elif choice == "4":

            plt.figure(figsize=(7,5))
            sns.histplot(df["Tenure"], bins=30, kde=True)
            plt.title("Tenure Distribution")
            plt.xlabel("Tenure")
            plt.show()


        elif choice == "5":

            plt.figure(figsize=(7,5))
            sns.countplot(x='Internet Service', hue='Churn', data=graph_df)
            plt.title("Internet Service vs Churn")
            plt.show()


        elif choice == "6":

            plt.figure(figsize=(8,5))
            sns.countplot(x='Payment Method', hue='Churn', data=graph_df)
            plt.xticks(rotation=20)
            plt.title("Payment Method vs Churn")
            plt.show()


        elif choice == "7":

            plt.figure(figsize=(10,8))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm",fmt=".2f")
            plt.title("Correlation Heatmap")
            plt.show()


        elif choice == "8":
            break

        else:
            print("Invalid Choice")

def model_analysis():

    print("\n========== MODEL ANALYSIS ==========\n")

    # Comparison Table
    print("Model Comparison:\n")
    print(comparison)

    # Best Model
    print("\nBest Model")
    print("-----------")
    print("Model :", best["Model"])
    print("Accuracy :", best["Accuracy"])

    # Cross Validation
    print("\nCross Validation")
    print("----------------")
    print("Scores :", scores)
    print("Average Accuracy :", scores.mean())

    # Grid Search
    print("\nGrid Search CV")
    print("--------------")
    print("Best Parameters :", grid_search.best_params_)
    print("Best F1 Score :", grid_search.best_score_)

    # Classification Reports
    print("\nDecision Tree Report")
    print("--------------------")
    print(classification_report(Y_test, test_predictions))

    print("\nLogistic Regression Report")
    print("--------------------------")
    print(classification_report(Y_test, pred_test))

    print("\nRandom Forest Report")
    print("--------------------")
    print(classification_report(Y_test, pre_test))

    # Learning Curve
    plt.figure(figsize=(8,5))

    plt.plot(train_sizes,
             train_mean,
             marker='o',
             label="Training Score")

    plt.plot(train_sizes,
             test_mean,
             marker='o',
             label="Validation Score")

    plt.title("Learning Curve")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.show()

#model_analysis()

def predict_customer(senior_citizen,partner,dependents,tenure,internet_service,online_security,online_backup,device_protection,tech_support,
    streaming_tv,streaming_movies,contract,paperless_billing,payment_method,monthly_charges,total_charges
):

    customer = pd.DataFrame([{
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure": tenure,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges
    }])

    prediction = model_rf.predict(customer)

    if prediction[0] == 1:
        return "Customer is likely to Churn."
    else:
        return "Customer is likely to Stay."
    
if __name__ == "__main__":
    
 while True:

    print("\n========== CUSTOMER CHURN PREDICTION ==========")
    print("1. Predict Customer Churn")
    print("2. Graphical Analysis")
    print("3. Model Analysis")
    print("4. Exit")

    choice = input("Enter your choice: ")

    if choice == "1":

        senior_citizen = int(input("Senior Citizen (0/1): "))
        partner = int(input("Partner (0/1): "))
        dependents = int(input("Dependents (0/1): "))
        tenure = int(input("Tenure: "))
        internet_service = int(input("Internet Service: "))
        online_security = int(input("Online Security (0/1): "))
        online_backup = int(input("Online Backup (0/1): "))
        device_protection = int(input("Device Protection (0/1): "))
        tech_support = int(input("Tech Support (0/1): "))
        streaming_tv = int(input("Streaming TV (0/1): "))
        streaming_movies = int(input("Streaming Movies (0/1): "))
        contract = int(input("Contract: "))
        paperless_billing = int(input("Paperless Billing (0/1): "))
        payment_method = int(input("Payment Method: "))
        monthly_charges = float(input("Monthly Charges: "))
        total_charges = float(input("Total Charges: "))

        result = predict_customer(
            senior_citizen,
            partner,
            dependents,
            tenure,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies,
            contract,
            paperless_billing,
            payment_method,
            monthly_charges,
            total_charges
        )

        print("\nPrediction :", result)

    elif choice == "2":
        graphical_analysis()

    elif choice == "3":
        model_analysis()

    elif choice == "4":
        print("Exiting Program...")
        break

    else:
        print("Invalid Choice! Please try again.")"""
