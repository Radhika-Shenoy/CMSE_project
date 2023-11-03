
import itertools
import random
import pandas as pd
import streamlit as st 
import plotly as plt
from pywaffle import Waffle
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns
import plotly.express as px
import plotly.subplots as sp
import numpy as np
import hiplot as hip
import networkx as nx
from streamlit_option_menu import option_menu
from PIL import Image

#Load the dataset
flight_df = pd.read_csv('passenger_exp_train.csv')

#Imputation is performed on the Arrival Delay in Minutes column where N/A values is filled with zero
#The assumption that the people who have filled the survey didn't experience arrival delay
flight_df['Arrival Delay in Minutes'] = flight_df['Arrival Delay in Minutes'].fillna(0) 
def categorize_age(age):
        if 3 < age <= 14:
            return 'Children: 3yrs - 14yrs'
        elif 14 < age <= 60:
            return 'Adults: 14yrs - 60 yrs'
        else:
            return 'Seniors: greater than 60yrs'
flight_df['Age_Group'] = flight_df['Age'].apply(categorize_age)
st.set_page_config(layout="wide")

#Inclusions for background image
image_url = '''
    <style>
    [data-testid="stAppViewContainer"] {
    background-image: url('https://images.unsplash.com/photo-1501630834273-4b5604d2ee31?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwxfDB8MXxyYW5kb218MHx8fHx8fHx8MTY4MTcwMTE1Mg&ixlib=rb-4.0.3&q=80&utm_campaign=api-credit&utm_medium=referral&utm_source=unsplash_source&w=1080');
    background-size: cover;
    background-repeat: no-repeat;
    }
    </style>
    '''
st.markdown(image_url, unsafe_allow_html=True)


#Creating side panel with navigation options for admin
with st.sidebar:
    selected = option_menu( menu_title="FlyHigh Admin navigation options",
    options=["Home",'Overall Airline Stats', "Analysis Page", "Advanced Analysis","Analysis Results","Manager Feedback"],
    icons=['house', 'bar-chart', "list-task", 'gear','download','envelope'], 
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    },
    menu_icon="cast",
    default_index=0)

#Sidepanel functionalities
#Functionalities performed when user is on HOMEPAGE
if selected == 'Home':
    st.markdown('<h1 style="color:black;font-size:34px;">🛫FlyHigh Airlines: Rise beyond the clouds⛅</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:black;font-size:20px;"><em>As a brand, we are dedicated to crafting unforgettable experiences for every passenger that takes to the skies with us, ensuring every moment is filled with delight and wonder.</em></h3>', unsafe_allow_html=True)
    
    image = Image.open('/Users/radhikavittalshenoy/Downloads/homepage_image.jpg')
    st.image(image, width=600)

    st.markdown('<em>Welcome to FlyHigh Airlines Home page! At FlyHigh Airlines, we are dedicated to providing all our flyers with an exceptional and comfortable flying experience. Our brand strives to offer the best-in-class services, ensuring their satisfaction is our top priority. With a focus on efficient operations and unparalleled customer service, we aim to make your journey with us as smooth and enjoyable as possible.Our Dashboard provides a comprehensive overview of key performance indicators, customer feedback, and operational insights. From passenger satisfaction ratings to on-time performance statistics, this page offers a holistic view of our airline performance. With an emphasis on safety, comfort,entertainmen and convenience, we constantly strive to enhance our services and meet the evolving needs of our valued passengers.</em>', unsafe_allow_html=True)
    
#Inclusions when navigated to the Overall Airline stats page
elif selected == "Overall Airline Stats":
    st.markdown('<h2 style="color:black;">FlyHigh Stats Gallery</h1>', unsafe_allow_html=True)
    st.markdown('<em>Explore the Comprehensive Insights of FlyHigh Airlines Brand Performance. Dive into the analysis of crucial factors like age and gender, evaluating the satisfaction levels of our esteemed past passengers. Engage with captivating visualizations and uncover the story behind our passengers experiences!</em>', unsafe_allow_html=True)

    #extract counts of the 'satisfaction' feature and use these counts for creating a pie chart with labels and values
    customer_type_counts = flight_df['satisfaction'].value_counts()
    values = customer_type_counts.values
    labels = customer_type_counts.index
    colors = ['#BFEFFF', '#1E90FE']

    
    fig = go.Figure(data=go.Pie(values=values, labels=labels, pull=[0.01, 0.04, 0.01, 0.05], hole=0.45, marker_colors=colors))

    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)

    fig.add_annotation(x=0.5, y=0.5, text='Satisfaction',
                    font=dict(size=18, family='Verdana', color='black'), showarrow=False)
    st.markdown('<h3 style="color:black;">Overall passenger satisfaction at FlyHigh Airlines</h3>', unsafe_allow_html=True)

    #Gender wise passenger percentage is computed
    gender_counts = flight_df['Gender'].value_counts()
    gender_percentage = (gender_counts / len(flight_df)) * 100
    fig2 = plt.figure(
        FigureClass=Waffle,
        rows=5,
        figsize=(8, 6),
        values=gender_percentage,
        labels=[f"Female ({gender_percentage['Female']:.2f}%)", f"Male ({gender_percentage['Male']:.2f}%)"],  # legend labels with percentages
        colors=["#FF82AB", "#1E90FE"],
        icons=['female', 'male'],
        legend={'loc': 'lower center',
                'bbox_to_anchor': (0.5, -0.5),
                'ncol': len(gender_counts),
                'framealpha': 0.5,
                'fontsize': 12
                },
        icon_size=15,
        icon_legend=True,
    )   
    
    #Plot 1 - Pie chart for satisfaction
    st.plotly_chart(fig, use_container_width=True)
    
    #Plot 2 - Waflle chart for gender distribution
    st.markdown('<h3 style="color:black;font-size:25px;">Overall FlyHigh Airlines Gender Distribution pattern</h3>', unsafe_allow_html=True)
    st.pyplot(fig2)

    #Plot 3 - Sunburst chart w=inidcating the satisfaction of each customer type
    st.markdown('<h3 style="color:black;font-size:25px;">Proportion of Satisfaction based on Customer Type</h3>', unsafe_allow_html=True)
    fig = px.sunburst(flight_df,path=["Customer Type","satisfaction"],template="plotly")
    st.plotly_chart(fig)
    
    #Function to categorize travellers based on age group
    

    # Applying categorize_age() to 'Age' column in flight_df to create a new column 'Age_Group' based on categorized age groups
    
    age_counts = flight_df['Age_Group'].value_counts()
    
    colors = ['#1E90FF', '#98F5FF']

    #Plot 4 - Pie chart showing each Age category distribution
    st.markdown('<h3 style="color:black;font-size:25px;">Traveller Age category distribution</h3>', unsafe_allow_html=True)
    fig = px.pie(
        values=age_counts.values,
        names=age_counts.index,
        color_discrete_sequence=colors
    )

    fig.update_traces(textinfo='percent', pull=[0, 0], textfont=dict(size=18),insidetextorientation='auto') 

    fig.update_layout(
        showlegend=True,
        #title_font=dict(size=15),
        width=550,
        height=500
    )
    st.plotly_chart(fig)


    #Plot 5 - Bar plot for age group distribution
    st.markdown('<h3 style="color:black;font-size:25px;">Age Group Analysis</h3>', unsafe_allow_html=True)
    
    #Count the occurrences of each age in the 'Age' column and then resetting the index to obtain the count of each unique value in the 'Age' column
    age_count = flight_df['Age'].value_counts().reset_index()
    fig = px.bar(
        age_count,
        x='Age',
        y='count',
        #title='Age Group Analysis',
        labels={'Count': 'Number of Customers'},
        color='Age',
        color_discrete_sequence=px.colors.sequential.Blues[::-1], 
    )

    fig.update_traces(
        text=age_count['count'], 
        textposition='outside', 
        marker=dict(line=dict(color='#000000', width=1)), 
    )

    fig.update_layout(
        xaxis_title='Age Group',
        yaxis_title='Number of Customers',
        font=dict(size=12),
        #title_font=dict(size=16),
        showlegend=False,
        plot_bgcolor='#FFFFFF',
        margin=dict(l=25, r=20, t=100, b=30),
    )
    st.plotly_chart(fig)

    
    st.markdown('<h3 style="color:black;font-size:25px;">Average Age by Travel Purpose</h3>', unsafe_allow_html=True)
    #Plot 6 - Bar plot for a passenger's Type of Travel distribution
    average_age_by_purpose = flight_df.groupby('Type of Travel')['Age'].mean().reset_index()
    fig = px.bar(average_age_by_purpose, x='Type of Travel', y='Age',
            labels={'Age': 'Average Age', 'Type of Travel': 'Travel Purpose'},
            color='Type of Travel', 
            color_discrete_sequence=['#66B2FF', 'grey'], 
            )

    fig.update_xaxes(categoryorder='array', title_text='Travel Purpose')
    st.plotly_chart(fig)

# ANALYSIS PAGE
elif selected == 'Analysis Page':
    st.markdown('<h2 style="color:black";font-size:20px;"> Data Insights Discovery Center<h2>', unsafe_allow_html=True)
    st.markdown('<em>Uncover the perfect blend of journey elements that craft an unforgettable travel experience, leveraging advanced data visualization tools to discern evolving trends and pivotal moments in our passengers journeys. </em>', unsafe_allow_html=True)
    selected_features = []
    #Columns 'Unnamed: 0' and 'id' are dropped as they have insignificant data
    flight_df = flight_df.drop(columns=['Unnamed: 0', 'id'])
    columns = list(flight_df.columns)
    categorical_data = list(set(flight_df.columns) - set(flight_df._get_numeric_data().columns))
    non_categorical_data = list(set(columns) - set(categorical_data))   
    plot_options = st.radio("Plot options",('Scatter plot', 'Histogram', 'Pie chart'))   
    if plot_options == 'Scatter plot':
        feature_options = st.multiselect('Select 2 Features', non_categorical_data, key="multiselect") 
        if len(feature_options)!=2:
            st.warning('Please select 2 features for viewing graphical visualization')
        if len(feature_options) ==2:
            categorical_feature = st.selectbox('Select the feature against which you want to analyse the already selected feature', categorical_data)
            if categorical_feature:
                # Plot 7 - Scatter plot based on selected features 
                st.pyplot(sns.scatterplot(data=flight_df, x=feature_options[0], y=feature_options[1], hue=categorical_feature).figure)
            
    elif plot_options == 'Histogram':
        feature_options = st.multiselect('Select a Feature', non_categorical_data, key="multiselect") 
        if len(feature_options) >1:
            st.warning('Please select one feature for viewing graphical histogram visualization')
        if len(feature_options) == 1:
            # Selection of feature provided for user 
            categorical_feature = st.selectbox('Select the feature against which you want to analyse the already selected feature', categorical_data)
            if categorical_feature:
                fig=px.histogram(flight_df,x=feature_options[0],color=flight_df[categorical_feature],title=f"{feature_options[0]} vs {categorical_feature}",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
                fig.update_layout(template="plotly")
                fig.update_layout(title_font_size=30)
                #Plot 8 - Histogram based on features selected
                st.plotly_chart(fig)
    
    elif plot_options == 'Pie chart':
        categorical_feature = st.selectbox('Select a categorical features for visualization', categorical_data) 
        non_categorical_feature = st.selectbox('Select a non-categorical features for visualization', non_categorical_data)    
        plt.figure(figsize=(20,10))
        fig=px.pie(values=flight_df[non_categorical_feature], names=flight_df[categorical_feature])
        #Plot 9 - Pie chart
        st.plotly_chart(fig, theme="streamlit")

#ADVANCED ANALYSIS
elif selected == 'Advanced Analysis':
    c = 0
    st.markdown('<h2 style="color:black";font-size:20px;"> Immersive Data Exploration Hub<h2>', unsafe_allow_html=True)
    st.markdown('<em>The FlyHigh R&D team can leverage these intricate passenger experience insights to uncover the depths of customer satisfaction, fostering innovative ideas to elevate our services and strengthen our brand reputation for unparalleled joyous travel experiences. </em>', unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<em>Disclaimer: Login access only for a member on the R&D team</em>', unsafe_allow_html=True)
        st.markdown('<em>Please login to view high level visualization</em>', unsafe_allow_html=True)
        with st.form(key='login_form'):
            st.subheader('Login Credentials')
            username = st.text_input('**Enter your FlyHigh admin userID:**')
            st.markdown('Default username: adminuser')
            password = st.text_input('**Enter your FlyHigh admin password:**', type='password')
            st.markdown('Default password: adminpwd')
            checkbox_val = st.checkbox("**I am an authorized FlyHigh member from the R&D team and wish to view the high dimensional representation of features!**")
            login_button = st.form_submit_button('Login')
        if login_button:
            if username == 'adminuser' and password == 'adminpwd':               
                if not checkbox_val:
                    st.write("Please agree that you are an authorized FlyHigh member prior to login")
                if checkbox_val:
                    st.success("Login successful!")
                    c = c+1

            else:
                st.error("Invalid username/ password")
    if c ==1:
        #Hi-dimensional plot for the passenger satisfaction factors is plotted
        df_f = flight_df.select_dtypes(include=[np.number])
        cols = flight_df[['Age', 'Gender', 'Customer Type', 'Type of Travel', 'Flight Distance','satisfaction']]
        hiplot_exp = hip.Experiment.from_dataframe(cols)
        hiplot_html = hiplot_exp.to_html()
        st.components.v1.html(hiplot_html, width=800, height=1300)

#ANALYSIS RESULTS
elif selected == "Analysis Results":
    st.markdown('<h2 style="color:black";font-size:20px;">Key Takeaways</h2>', unsafe_allow_html=True)
    st.markdown("""
    **This summary report represents the analysis of overall passenger satisfaction parameters and the impact of each factor on their travel experience. The displayed data is subject to weekly updates, reflecting real-time survey data collected by our R&D team.**
    """, unsafe_allow_html=True)
    st.markdown("""
    <style>
    .custom_bullet {
        list-style-type: none;
    }
    </style>

    <div class="custom_bullet">
        <p><b>&rarr; A major concern for the airline is that 56.7% of passengers are either neutral or dissatisfied.</p>
        <p><b>&rarr; Passenger satisfaction does not appear to be influenced by gender. However, it seems to be affected by factors such as age, flight delays, and specific services, among other variables.</b></p>
        <p><b>&rarr; The majority of the passengers appear to be neutral or unsatisfied, with the majority of the travelers being female.</b></p>
        <p><b>&rarr; Most of the travelers fall in the age group of 22yrs -50yrs. People who are satisfied tend to be in their 40s to 60s. Most of the unhappy passengers are between the ages of 20 and 40.</b></p>
        <p><b>&rarr; When it comes to travelers on business trips, they are typically a little older than those traveling for personal reasons.</b></p>
        <p><b>&rarr; The distance range of passengers travelling for Personal reasons travel uptil a distance range of 2500 miles whereas for Business travel purposes they travel farther.</b></p>
        <p><b>&rarr; Airlines' services are rated a 4 out of 5, with luggage handling and in-flight service being the best and in-flight Wi-Fi service being the worst.</b></p>
    </div>
    """, unsafe_allow_html=True)

#MANAGER FEEDBACK
elif selected == "Manager Feedback":
    st.markdown('<h2 style="color:black";font-size:20px;">Suggestion from the Manager to the Research and Development team of FlyHigh</h3>', unsafe_allow_html=True)

    input_text = st.text_input("Type in you're feedback here")
    arrow_clicked = st.button("Share")
    # Check if the arrow button is clicked
    if arrow_clicked and input_text:
        # Perform actions based on the input text
        st.success(f"Suggestion shared succefully!")
    if arrow_clicked and not input_text:
        st.error(f"Please enter a suggestion")

st.sidebar.markdown(f'<a href="https://flyhighproject-7fmy8ekgwxnxu8d4cwyqhr.streamlit.app"><button>Logout</button> </a>', unsafe_allow_html=True)


#Standard scalar
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
LE = LabelEncoder()
print(flight_df)
flight_df['Class'] = LE.fit_transform(flight_df['Class'])
flight_df['satisfaction'] = LE.fit_transform(flight_df['satisfaction'])
flight_df['Type of Travel'] = LE.fit_transform(flight_df['Type of Travel'])
flight_df['Gender'] = LE.fit_transform(flight_df['Gender'])
flight_df['Customer Type'] = LE.fit_transform(flight_df['Customer Type'])

features = flight_df.drop(columns=['satisfaction','Age_Group'],axis=1)
target = flight_df['satisfaction']


X_train,X_test,y_train,y_test = train_test_split(features,target,test_size=0.2,random_state=42)

sd = StandardScaler()
X_train_scaled = sd.fit_transform(X_train)
X_test_scled = sd.transform(X_test)

# X_train_scaled has the features scaled to one scale for further predictive modelling for Phase 2 of the project

        



            
        
        
        
