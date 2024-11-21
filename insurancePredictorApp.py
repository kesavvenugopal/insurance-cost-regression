# necessary imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV

# dataset is accessed via 'df'
df=pd.read_csv("medical_insurance.csv")

# outliers are plotted in the Jupyter Notebook. Refer it for more info

# eliminating the outliers using IQR rule
Q1=df['bmi'].quantile(0.25)
Q3=df['bmi'].quantile(0.75)

IQR=Q3-Q1
lower_bound= Q1-1.5*IQR
upper_bound= Q3+1.5*IQR
# print(lower_bound,upper_bound)

df_filtered=df[(df['bmi']>=lower_bound) & (df['bmi']<47)] # upper bound = 47 removes all outliers in one step. If not, this whole step has to be
                                                          # repeated 2 or more times.

# print(df_filtered.shape)
# print(df.shape)

# renaming the filtered dataset for easier management
df=df_filtered

# checking for null values
# print(df.isnull().sum())

#splitting the feature and target variables
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
# print(x)
# print(y)

# initialising the label encoders
labEnc_sex=LabelEncoder()
labEnc_smoker=LabelEncoder()
labEnc_region=LabelEncoder()

# label encoding categorical columns
x[:, 1] = labEnc_sex.fit_transform(x[:, 1])  # Gender: Female 0, Male 1
x[:, 4] = labEnc_smoker.fit_transform(x[:, 4])  # Smoker: Yes 1, No 0
x[:, 5] = labEnc_region.fit_transform(x[:, 5])  # Region: Northeast 0, Northwest 1, Southeast 2, Southwest 3
# print(x)

# initialising ColumnTransformer with OneHotEncoder algorithm
ct=ColumnTransformer(
    transformers=[
        ('onehot',OneHotEncoder(drop='first'),[1,4,5]) # Dropping the dummy variable column(1st column after each transform) using drop='first'
    ],
    remainder='passthrough'
)

# fitting and transforming the feature set
x=ct.fit_transform(x)
# print(x[0])

# splitting the training data and testing data
x_test,x_train,y_test,y_train=train_test_split(
    x,y,test_size=0.25,random_state=2
)

# creating an instance of StandardScaler instance and fitting and transforming the feature set
ss_x = StandardScaler()
scaled_x_train = ss_x.fit_transform(x_train)
scaled_x_test = ss_x.transform(x_test)

# creating a function for model fitting, training and evaluation
def model_evaluation(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    r2 = metrics.r2_score(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    
    return r2, mae, rmse


# models dictionary
# the hyperparameters are obtained after HyperParameter Tuning uing GridSearchCV. Refer the Jypyter Notebook for the tuning code
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree Regression": DecisionTreeRegressor(criterion='squared_error', max_depth=10, max_features= None, min_samples_leaf= 4, min_samples_split= 10, splitter='best'),
    "Random Forest Regression": RandomForestRegressor(bootstrap=False, criterion='absolute_error', max_depth=10, max_features='log2', min_samples_leaf=1, min_samples_split=2, n_estimators=200),
    "KNN Regression": KNeighborsRegressor(algorithm='auto', leaf_size=20, metric='euclidean', n_neighbors=15, weights='distance'),
    "SVR Regression": SVR(C=100, degree=3, epsilon=0.5, gamma='scale', kernel='linear')
}

# calling the model evaluation function. for KNN and SVR algorithms, scaled featureset are used to improve the accuracy
results = []
for model_name, model in models.items():
    if model_name == "KNN Regression" or model_name == "SVR Regression":
        r2, mae, rmse = model_evaluation(model, scaled_x_train, scaled_x_test, y_train, y_test)
    else:
        r2, mae, rmse = model_evaluation(model, x_train, x_test, y_train, y_test)
    
    results.append({
        "Model": model_name,
        "R-squared": r2,
        "Mean Absolute Error": mae,
        "Root Mean Squared Error": rmse
    })

# printing the model evaluation dataset
results_df = pd.DataFrame(results)
# print(results_df.to_string(index=False))

# here we can observe that Random Forest algorithm has the better stats. Hence for custom input for insurance cost prediction, Random
# Forest Algorithm is used

# in program custom input
x_custom = np.array([[23, 'male', 27.8, 0, 'yes', 'southeast']], dtype=object)

x_custom[:, 1] = labEnc_sex.transform(x_custom[:, 1])
x_custom[:, 4] = labEnc_smoker.transform(x_custom[:, 4])
x_custom[:, 5] = labEnc_region.transform(x_custom[:, 5])

x_custom = ct.transform(x_custom)
y_custom = models['Random Forest Regression'].predict(x_custom)
# print(print("Insurance cost prediction for custom input data is:",y_custom[0].round(2)))

# defining a function to get the inputs from the user.
def get_input():
    age = int(input("Enter the age: "))
    bmi = float(input("Enter BMI: "))
    children = int(input("Enter no. of children: "))
    
    # infinite loops breaks only if valid inputs are received
    while True:
        print("\nSelect Gender")
        print("1: Female")
        print("2: Male")
        gender_choice = input("Enter either 1 or 2: ")
        if gender_choice == '1':
            gender = 'female'
            break
        elif gender_choice == '2':
            gender = 'male'
            break
        else:
            print("Invalid input. Please enter 1 for Female or 2 for Male.")

    while True:
        print("\nAre you a smoker?")
        print("1: Yes")
        print("2: No")
        smoker_choice = input("Enter 1 for Yes or 2 for No: ")
        if smoker_choice == '1':
            smoker = 'yes'
            break
        elif smoker_choice == '2':
            smoker = 'no'
            break
        else:
            print("Invalid input. Please enter 1 for Yes or 2 for No.")
    
    while True:
        print("\nSelect region:")
        print("1: Northeast")
        print("2: Northwest")
        print("3: Southeast")
        print("4: Southwest")
        region_choice = input("Enter 1, 2, 3, or 4 for the corresponding region: ")
        if region_choice == '1':
            region = 'northeast'
            break
        elif region_choice == '2':
            region = 'northwest'
            break
        elif region_choice == '3':
            region = 'southeast'
            break
        elif region_choice == '4':
            region = 'southwest'
            break
        else:
            print("Invalid input. Please enter a number from 1 to 4 for the region.")

    return np.array([[age, gender, bmi, children, smoker, region]], dtype=object)

# calling the function to get user input.
x_user = get_input()
cols=['Age','Gender','BMI','No. of Children','Smoker','Region']
userDF=pd.DataFrame(x_user,columns=cols)
print('\nUser Input Data\n',userDF.to_string(index=False))

# encoding and transforming the input values
x_user[:,1]=labEnc_sex.transform(x_user[:,1])
x_user[:,4]=labEnc_smoker.transform(x_user[:,4])
x_user[:,5]=labEnc_region.transform(x_user[:,5])
x_user=ct.transform(x_user)

# predicting the insurance cost
y_user=models['Random Forest Regression'].predict(x_user)

# printing the prediction
print("Insurance cost prediction for user input data is:",y_user[0].round(2))