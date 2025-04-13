import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data):
    # Separate features and target variable
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model
    y_pred = model.predict(X_test)
    print('Accuracy of our model:', accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))
    return model, scaler


def get_clean_data():
    # Load the dataset
    data = pd.read_csv("data/data.csv")
    
    # Drop unnecessary columns
    data = data.drop(['Unnamed: 32', 'id'], axis=1)

    # Map target labels to binary values
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data


def main():
    # Clean and prepare data
    data = get_clean_data()

    # Create and evaluate the model
    model, scaler = create_model(data)

    with open('model/model.pkl','wb') as f:
        pickle.dump(model,f)
    
    with open('model/scaler.pkl','wb') as f:
        pickle.dump(scaler,f)


if __name__ == '__main__':
    main()
