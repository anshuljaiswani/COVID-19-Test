import dash
from dash_bootstrap_components._components.Jumbotron import Jumbotron
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Output, Input
from dash_html_components.Div import Div
from dash_html_components.Img import Img
import plotly.express as px
import dash_bootstrap_components as dbc
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_curve, auc
from sklearn import preprocessing
from sklearn import utils
from numpy import genfromtxt, left_shift, number
from sklearn import linear_model
from dash.dependencies import Input, Output, State
sns.set()
import pandas as pd
import csv
import numpy as np
df=pd.read_csv('./coronadb.csv')
print(df.head())
Y = df['Result']
X = df[['Fev','DC','Tir','A&P','ST','Dia','Hea','T&S','Rash','Bre','Chest']]
lab_enc = preprocessing.LabelEncoder()
training_scores_encoded = lab_enc.fit_transform(Y)
lr=linear_model.LogisticRegression()
lr.fit(X,training_scores_encoded)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])


app.layout=html.Div([
dbc.Navbar(
    [
        html.A(
            dbc.Row(
                [
                    
                    dbc.Col(html.Img(src ="/assets/co4.jpg" , height="50px")),                    
                    dbc.Col(dbc.NavbarBrand("CoviTest", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
                
            ),
            
        ),
        
    ],
    color="black",
    dark=True,
),
dbc.Jumbotron( 
    [
        dbc.Container(
            [
                html.Div(className="con"),
                html.H1("COVID-19 Analaysis", className="display-1"),
                html.P(
                   "This site is use to predict the probability that you are Covid positive or not "
                   "on the basis of information given by you.",
                    className="lead",
                    
                ),
                dbc.Col(html.Img(src="/assets/co2.jpeg",height="300px",width="100%" )),
                

                
            ],
            fluid=True,
            
        )
    ],
    fluid=True,
),
dbc.Container([
  
    dbc.Row(
    [
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Full Name", html_for="example-email-grid",),
                    dbc.Input(
                        type="string",
                        id="example-email-grid",
                        placeholder="Full Name",
                    ),
                ]
            ),
            width=6,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Phone Number", html_for="example-password-grid"),
                    dbc.Input(
                        type="int",
                        id="example-password-grid",
                        placeholder="Phone Number",
                    ),
                ]
            ),
            width=6,
        ),
    ],
    form=True,style={'align':'left'},
),

 dbc.Row([
     dbc.Col(
     dbc.FormGroup(
    [
        dbc.Label("State", html_for="dropdown", style={ 'size' :'6px'}),
        dcc.Dropdown( 
            id="dropdown",
            options=[
                {"label": "Andhra Pradesh", "value": 1},
                {"label": "Arunachal Pradesh", "value": 2},
                {"label": "Assam", "value": 3},
                {"label": "Bihar", "value": 4},
                {"label": "Chandigarh", "value": 5},
                {"label": "Chandigarh", "value": 6},
                {"label": "Delhi", "value": 7},
                {"label": "Goa", "value": 8},
                {"label": "Gujarat", "value": 9},
                {"label": "Haryana", "value": 10},
                {"label": "Himachal Pradesh", "value": 11},
                {"label": "Jammu and Kashmir", "value": 12},
                {"label": "Jharkhand", "value": 13},
                {"label": "Karnataka", "value": 14},
                {"label": "Kerala", "value": 15},
                {"label": "Madhya Pradesh", "value": 16},
                {"label": "Maharashtra", "value": 17},
                {"label": "Manipur", "value": 18},
                {"label": "Meghalaya", "value": 19},
                {"label": "Mizoram", "value": 20},
                {"label": "Nagaland", "value": 21},
                {"label": "Odisha", "value": 22},
                {"label": "Punjab", "value": 23},
                {"label": "Rajasthan", "value": 24},
                {"label": "Sikkim", "value": 25},
                {"label": "Tamil Nadu", "value": 26},
                {"label": "Telangana", "value": 27},
                {"label": "Tripura", "value": 28},
                {"label": "Uttar Pradesh", "value": 29},
                {"label": "Uttarakhand", "value": 30},
                {"label": "West Bengal ", "value": 31},
                
            ],
        ),
     
    ]
), width=12,
),  


]),

dbc.Row(
    [
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Age", html_for="example-email-grid",),
                    dbc.Input(
                        type="number",
                        
                        placeholder="Enter Age",
                    ),
                ]
            ),
            width=6,
        ),
        dbc.Col(
            dbc.FormGroup(
                [
                    dbc.Label("Body Temprature", html_for="example-password-grid"),
                    dbc.Input(
                        type="number",
                       
                        placeholder="Body Temprature",
                    ),
                ]
            ),
            width=6,
        ),

    ],
    form=True,
),

dbc.FormGroup(
    [
        dbc.Label("Fever", html_for="slider"),
        dcc.Slider(id="Fev", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Dry Cough", html_for="slider"),
        dcc.Slider(id="DC", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Tiredness", html_for="slider"),
        dcc.Slider(id="Tir", min=0, max=10, step=0.5, value=3  ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Aches & Pains", html_for="slider"),
        dcc.Slider(id="A&P", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Sore Throat", html_for="slider"),
        dcc.Slider(id="ST", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Diarrhoea", html_for="slider"),
        dcc.Slider(id="Dia", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Headache", html_for="slider"),
        dcc.Slider(id="Hea", min=0, max=10, step=0.5, value=3),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Loss of Taste or Smell", html_for="slider"),
        dcc.Slider(id="T&S", min=0, max=10, step=0.5, value=3),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("A rash on skin, or discolouration of fingers or toes", html_for="slider"),
        dcc.Slider(id="Rash", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Difficulty breathing or shortness of breath", html_for="slider"),
        dcc.Slider(id="Bre", min=0, max=10, step=0.5, value=3 ),
    ]
),
dbc.FormGroup(
    [
        dbc.Label("Chest pain or pressure", html_for="slider"),
        dcc.Slider(id="Chest", min=0, max=10, step=0.5, value=3 ),
    ]
),



dbc.Row(
        dbc.Col(html.H3("Percent",id="Result",
                        className='text-center mb-4 card-text  text-success border border-primary',style={'borderRadius':'22px','font_family':  "Courier New",'padding':'22px', 'left':'100px'}),
                width=12)
                
    ), 
 
]),
  

  ])
 
@app.callback(Output(component_id="Result",component_property="children"),
# The values correspnding to the three sliders are obtained by calling their id and value property
              [Input("Fev","value"), Input("DC","value"), Input("Tir","value"),
              Input("A&P","value"), Input("ST","value"), Input("Dia","value"), 
              Input("Hea","value"), Input("T&S","value"), Input("Rash","value"),
               Input("Bre","value"), Input("Chest","value")
              
              
              ])

# The input variable are set in the same order as the callback Inputs
def update_prediction(x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11):
    print(x1)
    L=[int(x1),int(x2),int(x3),int(x4),int(x5),int(x6),int(x7),int(x8),int(x9),int(x10),int(x11)]
    b=int(lr.predict([L]))
    if(b==9):
        x="90%"
        print("90%")
    elif(b==1):
        x="10%"
        print("10%")        
    elif(b==2):
        x="20%"
        print("20%")
    elif(b==3):
        x="30%"
        print("30%")    
    elif(b==4):
        x="40%"
        print("40%")
    elif(b==5):
        x="50%"
        print("50%")
    elif(b==6):
        x="60%"
        print("60%")
    elif(b==7):
        x="70%"
        print("70%")
    elif(b==8):
        x="80%"
        print("80%")                        
    else:
        x="0%"
        print("0%")   
    return "Precent: {}".format(x)


if __name__ == '__main__':
    app.run_server(debug = True)
