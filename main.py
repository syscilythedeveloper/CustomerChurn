import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import utils
from openai import OpenAI



client = OpenAI(base_url="https://api.groq.com/openai/v1",
                api_key=os.environ.get("GROQ_API_KEY"))


def load_model(filename):
  with open(filename, "rb") as file:
    return pickle.load(file)


xgboost_model = load_model('xgb_model.pkl')

naive_bayes_model = load_model('nb_model.pkl')

random_forest_model = load_model('rf_model.pkl')

decision_tree_model = load_model('dt_model.pkl')

svm_model = load_model('svm_model.pkl')

knn_model = load_model('knn_model.pkl')

voting_classifier_model = load_model('voting_classifier.pkl')

xgboost_SMOTE_model = load_model('xgb_model-SMOTE.pkl')

xgboost_featureEngineering_model = load_model('xgboost-featureEngineered.pkl')


def prepare_input(credit_score, location, gender, age, tenure, balance,
                  num_products, has_credit_card, is_active_member,
                  estimated_salary):
  input_dict = {
      'CreditScore': credit_score,
      'Age': age,
      'Tenure': tenure,
      'Balance': balance,
      'NumOfProducts': num_products,
      'HasCrCard': has_credit_card,
      'IsActiveMember': int(is_active_member),
      'EstimatedSalary': estimated_salary,
      'Geography_France': 1 if location == "France" else 0,
      'Geography_Germany': 1 if location == "Germany" else 0,
      'Geography_Spain': 1 if location == "Spain" else 0,
      'Gender_Male': 1 if gender == "Male" else 0,
      'Gender_Female': 1 if gender == "Female" else 0,
  }
  input_df = pd.DataFrame([input_dict])
  return input_df, input_dict


def make_predictions(input_df, input_dict):
  probabilities = {
      'Random Forest': random_forest_model.predict_proba(input_df)[0][1],
      'K-Nearest Neighbors': knn_model.predict_proba(input_df)[0][1],
  }

  avg_probability = np.mean(list(probabilities.values()))

  col1, col2 = st.columns(2)

  with col1:
    fig = utils.create_gauge_chart(avg_probability)
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"The cystomer has a {avg_probability:.2%} probability of churning")

  with col2: 
    fig_probs = utils.create_model_probability_chart(probabilities)
    st.plotly_chart(fig_probs, use_container_width=True)

  st.markdown("### Model Probabilities")
  for model, prob in probabilities.items():
    st.write(f"{model} {prob}")
  st.write(f"Average Probability: {avg_probability}")
  return avg_probability


def explain_prediction(probability, input_dict, surname):
  prompt = f"""
  You are an expert data scientist at a bank, where you specialize in interpreting and explaining predictions of machine learning models. 

  Your machine learning model has predicted that a cutstomer names {surname} has a {round(probability*100, 1)}% probability of curching, based on the inofmration provided below. 

  Here is the customer's information:
  {input_dict}

  Here are the machine learning model's top 10 most i portant features for predicting churn: 
  
Feature - 	Importance
0	CreditScore	- 0.035005
1	Age	- 0.109550
2	Tenure	- 0.030054
3	Balance - 	0.052786
4	NumOfProducts - 	0.323888
5	HasCrCard	- 0.031940
6	IsActiveMember - 0.164146
7	EstimatedSalary - 	0.032655
8	Geography_France - 	0.046463
9	Geography_Germany	- 0.091373
10	Geography_Spain - 	0.036855
11	Gender_Female	- 0.045283
12	Gender_Male	- 0.000000

  {pd.set_option('display.max_columns' , None)}

  Here are summary statisti s for churned customers: 
  {df[df['Exited'] ==1].describe()}

  If the customer has over 40% risk of churning, you should explain why they are at risk of churning. If the customer has less than 40% risk of churning, you should explain why they are at less risk of churning. Each response should be between 3-5 sebtenecs. Base explantion on customer's info, the summary statistics of churned and non churned customers, and the feature importances provided. 

  Don't mention the probability of churning or the machine learning model or say anything like, "Based on the machhine learning model's predicition nd top 10 most important features. Just explain the prediction. 
  
  """
  print("Explanation prompt: ", prompt)

  raw_response = client.chat.completions.create(model="llama-3.2-3b-preview",
                                                messages=[{
                                                    "role": "user",
                                                    "content": prompt
                                                }])

  return raw_response.choices[0].message.content


def generate_email(probability, input_dict, explanations, surname):
  prompt = f"""
  You are a manager at HS Bank. You are repsonsible for ensuring customers stay with the bank and are incentiviized with various offers.  
  
  You noticed a customer named {surname} has a {round(probability*100, 1)}% probability of churning. 

  Here's the customer's information: {input_dict}
  Here are the customer's explanations: {explanations}

  Generate an email to the customer based on their information asking them to stay if they are at risk of churning, or offering them incentives so that they become more loyal to the bank. 

  List out incentives to stay in bullet point format with a blank line between each incentive. Don't mention the proability of churning or the machine learning model to the customer. Use "Your Friends at HS Bank" as the signature
  
  """
  raw_response = client.chat.completions.create(
      model="llama-3.2-3b-preview",  # or another valid OpenAI model name
      messages=[{
          "role": "user",
          "content": prompt
      }],
  )
  print("\n\nEMAIL Prompt: ", prompt)
  return raw_response.choices[0].message.content


st.title("Customer Churn Prediction")

df = pd.read_csv("churn.csv")

customers = [
    f"{row['CustomerId']} - {row['Surname']}" for _, row in df.iterrows()
]

selected_customer_option = st.selectbox("Select a customer", customers)

if selected_customer_option:
  selected_customer_id = int(selected_customer_option.split(" - ")[0])

  print("selected customer id", selected_customer_id)

  selected_customer_surname = selected_customer_option.split(" - ")[1]
  print("selected customer surname", selected_customer_surname)

  selected_customer = df.loc[df['CustomerId'] == selected_customer_id].iloc[0]

  print("Selected Customer", selected_customer)

  col1, col2 = st.columns(2)

  with col1:
    credit_score = st.number_input('Credit Score',
                                   min_value=300,
                                   max_value=850,
                                   value=int(selected_customer['CreditScore']))

    location = st.selectbox("Location", ["Spain", "France", "Germany"],
                            index=["Spain", "France", "Germany"
                                   ].index(selected_customer['Geography']))

    gender = st.radio("Gender", ["Male", "Female"],
                      index=0 if selected_customer['Gender'] == "Male" else 1)

    age = st.number_input("Age",
                          min_value=18,
                          max_value=100,
                          value=int(selected_customer['Age']))

    tenure = st.number_input("Tenure (years)",
                             min_value=0,
                             max_value=50,
                             value=int(selected_customer['Tenure']))

  with col2:
    balance = st.number_input("Balance",
                              min_value=0.0,
                              value=float(selected_customer['Balance']))

    num_products = st.number_input("Number of Products",
                                   min_value=1,
                                   max_value=10,
                                   value=int(
                                       selected_customer['NumOfProducts']))

    has_credit_card = st.checkbox("Has Credit Card",
                                  value=bool(selected_customer["HasCrCard"]))

    is_active_member = st.checkbox("Is Active Member",
                                   value=bool(
                                       selected_customer["IsActiveMember"]))

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=float(selected_customer['EstimatedSalary']))
  input_df, input_dict = prepare_input(credit_score, location, gender, age,
                                       tenure, balance, num_products,
                                       has_credit_card, is_active_member,
                                       estimated_salary)
  avg_probability = make_predictions(input_df, input_dict)

  explanation = explain_prediction(avg_probability, input_dict,
                                   selected_customer['Surname'])

  st.markdown("---")
  st.subheader("Explanation of Prediction")
  st.markdown(explanation)

  email = generate_email(avg_probability, input_dict, explanation,
                         selected_customer['Surname'])

  st.markdown("---")
  st.subheader("Personalized Email")
  st.markdown(email)
