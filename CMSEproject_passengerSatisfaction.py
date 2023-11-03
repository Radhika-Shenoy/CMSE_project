
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
flight_df = pd.read_csv('passenger_exp_train.csv')
#selection = st.sidebar.selectbox("Page View", ["Dashboard", "Analysis Page","Advanced Analysis"])
#st.set_page_config(page_title="Flyhigh", layout="wide")
st.set_page_config(layout="wide")


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




with st.sidebar:
    selected = option_menu( menu_title="FlyHigh Member navigation options",
    options=["Home",'Overall Airline Stats', "Analysis Page", "Advanced Analysis"],
    icons=['house', 'cloud-upload', "list-task", 'gear'], 
    styles={
        "container": {"padding": "0!important", "background-color": "#fafafa"},
        "icon": {"color": "black", "font-size": "20px"}, 
        "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "blue"},
    },
    menu_icon="cast",
    default_index=0)
if selected == 'Home':
    st.markdown('<h1 style="color:black;font-size:34px;">🛫FlyHigh Airlines: Rise beyond the clouds⛅</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:black;font-size:20px;"><em>As a brand, we are dedicated to crafting unforgettable experiences for every passenger that takes to the skies with us, ensuring every moment is filled with delight and wonder.</em></h3>', unsafe_allow_html=True)
    
    image = Image.open('homepage_image.jpg')
    st.image(image, width=600)

    st.markdown('<em>Welcome to FlyHigh Airlines Home page! At FlyHigh Airlines, we are dedicated to providing all our flyers with an exceptional and comfortable flying experience. Our brand strives to offer the best-in-class services, ensuring their satisfaction is our top priority. With a focus on efficient operations and unparalleled customer service, we aim to make your journey with us as smooth and enjoyable as possible.Our Dashboard provides a comprehensive overview of key performance indicators, customer feedback, and operational insights. From passenger satisfaction ratings to on-time performance statistics, this page offers a holistic view of our airline performance. With an emphasis on safety, comfort,entertainmen and convenience, we constantly strive to enhance our services and meet the evolving needs of our valued passengers.</em>', unsafe_allow_html=True)
    G = nx.Graph()

    nodes = ['Flyer Satisfaction', 'Baggage Handling', 'In flight service', 'Seat Comfort', 'Arrival Delay', 'Food and Drink','Entertatinment']
    edges = [('Flyer Satisfaction', 'Baggage Handling'), ('Flyer Satisfaction', 'In flight service'), ('Flyer Satisfaction', 'Seat Comfort'), ('Flyer Satisfaction', 'Arrival Delay'), ('Flyer Satisfaction', 'Food and Drink'), ('Flyer Satisfaction', 'Entertatinment')]
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Set the layout for the nodes
    pos = nx.spring_layout(G)

    # Draw the network graph
    options = {
        "node_size": 1800,
        "edge_color": "gray",
        "width": 1,
        "with_labels": True,
        "font_size": 5,
        "font_color": "k",
        "font_weight": "bold",
    }

    # Create figure without axes
    fig, ax = plt.subplots(figsize=(6, 4), facecolor='none')
    ax.set_axis_off()

    # Draw the network graph without borders and background box
    nx.draw_networkx(G, pos, ax=ax, **options)

    # Display the plot in Streamlit without the borders and background box
    #st.pyplot(fig, bbox_inches='tight', pad_inches=0)

elif selected == "Overall Airline Stats":
    st.markdown('<h2 style="color:black;">FlyHigh Stats Gallery</h1>', unsafe_allow_html=True)
    st.markdown('<em>Explore the Comprehensive Insights of FlyHigh Airlines Brand Performance. Dive into the analysis of crucial factors like age and gender, evaluating the satisfaction levels of our esteemed past passengers. Engage with captivating visualizations and uncover the story behind our passengers experiences!</em>', unsafe_allow_html=True)
    customer_type_counts = flight_df['satisfaction'].value_counts()

    values = customer_type_counts.values
    labels = customer_type_counts.index

    colors = ['#BFEFFF', '#1E90FE']

    fig = go.Figure(data=go.Pie(values=values, labels=labels, pull=[0.01, 0.04, 0.01, 0.05], hole=0.45, marker_colors=colors))

    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)

    fig.add_annotation(x=0.5, y=0.5, text='Satisfaction',
                    font=dict(size=18, family='Verdana', color='black'), showarrow=False)
    st.markdown('<h3 style="color:black;">Overall passenger satisfaction at FlyHigh Airlines</h3>', unsafe_allow_html=True)
    #fig.update_layout(title_text='Overall passenger satisfaction at FlyHigh Airlines', title_font=dict(size=15, family='Verdana'))


    flight_df = pd.read_csv('passenger_exp_train.csv')
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
    # Place the pie chart in the first column
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('<h3 style="color:black;font-size:25px;">Overall FlyHigh Airlines Gender Distribution pattern</h3>', unsafe_allow_html=True)
    st.pyplot(fig2)
    
    st.markdown('<h3 style="color:black;font-size:25px;">Proportion of Satisfaction based on Customer Type</h3>', unsafe_allow_html=True)
    fig = px.sunburst(flight_df,path=["Customer Type","satisfaction"],template="plotly")
    fig.update_layout(
       # title=dict(
        #    text="Satisfaction vs Customer Type",
         #   font=dict(size=15)
        
    )

    st.plotly_chart(fig)
    
    

    def categorize_age(age):
        if age <= 1:
            return 'Newborns: Less than 1yr'
        elif 1 < age <= 3:
            return 'Infants: 1yrs - 3yrs'
        elif 3 < age <= 14:
            return 'Children: 3yrs - 14yrs'
        elif 14 < age <= 60:
            return 'Adults: 14yrs - 60 yrs'
        else:
            return 'Seniors: greater than 60yrs'
    #column1,column2 = st.columns([1,1])
    #with column1:
    flight_df['Age_Group'] = flight_df['Age'].apply(categorize_age)
    age_counts = flight_df['Age_Group'].value_counts()

    colors = ['#1E90FF', '#98F5FF']
    st.markdown('<h3 style="color:black;font-size:25px;">Traveller Age category distribution</h3>', unsafe_allow_html=True)
    fig = px.pie(
        values=age_counts.values,
        names=age_counts.index,
        #title="Traveller Age category distribution",
        color_discrete_sequence=colors 
    )

    fig.update_traces(textinfo='percent+label', pull=[0.03, 0.02], textfont=dict(size=18),insidetextorientation='auto') 

    fig.update_layout(
        showlegend=True,
        #title_font=dict(size=15),
        width=500,
        height=500
    )
    st.plotly_chart(fig)

        #with column2:
    st.markdown('<h3 style="color:black;font-size:25px;">Age Group Analysis</h3>', unsafe_allow_html=True)
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

elif selected == 'Analysis Page':
    st.markdown('<h2 style="color:black";font-size:20px;"> Data Insights Discovery Center<h2>', unsafe_allow_html=True)
    st.markdown('<em>Uncover the perfect blend of journey elements that craft an unforgettable travel experience, leveraging advanced data visualization tools to discern evolving trends and pivotal moments in our passengers journeys. </em>', unsafe_allow_html=True)
    selected_features = []
    flight_df = flight_df.drop(columns=['Unnamed: 0', 'id'])
    columns = list(flight_df.columns)
    categorical_data = list(set(flight_df.columns) - set(flight_df._get_numeric_data().columns))
    non_categorical_data = list(set(columns) - set(categorical_data))   
    plot_options = st.radio("Plot options",('Scatter plot', 'Histogram', 'Pie chart'))    
    #plot_options = st.selectbox('Plot options', ('None','Scatter plot', 'Histogram', 'Pie chart'))
    if plot_options == 'Scatter plot':
        feature_options = st.multiselect('Select 2 Features', non_categorical_data, key="multiselect") 
        if len(feature_options)!=2:
            st.warning('Please select 2 features for viewing graphical visualization')
        if len(feature_options) ==2:
            categorical_feature = st.selectbox('Select the feature against which you want to analyse the already selected feature', categorical_data)
            if categorical_feature:
                st.pyplot(sns.scatterplot(data=flight_df, x=feature_options[0], y=feature_options[1], hue=categorical_feature).figure)
            
    elif plot_options == 'Histogram':
        feature_options = st.multiselect('Select a Feature', non_categorical_data, key="multiselect") 
        if len(feature_options) >1:
            st.warning('Please select one feature for viewing graphical histogram visualization')
        if len(feature_options) == 1:
            categorical_feature = st.selectbox('Select the feature against which you want to analyse the already selected feature', categorical_data)
            if categorical_feature:
                fig=px.histogram(flight_df,x=feature_options[0],color=flight_df[categorical_feature],title=f"{feature_options[0]} vs {categorical_feature}",
                        color_discrete_sequence=px.colors.qualitative.Vivid)
                fig.update_layout(template="plotly")
                fig.update_layout(title_font_size=30)
                st.plotly_chart(fig)
    
    elif plot_options == 'Pie chart':
        categorical_feature = st.selectbox('Select a categorical features for visualization', categorical_data) 
        non_categorical_feature = st.selectbox('Select a non-categorical features for visualization', non_categorical_data)    
        plt.figure(figsize=(20,10))
        fig=px.pie(values=flight_df[non_categorical_feature], names=flight_df[categorical_feature])
        st.plotly_chart(fig, theme="streamlit")

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
            st.markdown('Default username: user')
            password = st.text_input('**Enter your FlyHigh admin password:**', type='password')
            st.markdown('Default password: password')
            checkbox_val = st.checkbox("**I am an authorized FlyHigh member from the R&D team and wish to view the high dimensional representation of features!**")
            login_button = st.form_submit_button('Login')
        if login_button:
            if username == 'user' and password == 'password':               
                if not checkbox_val:
                    st.write("Please agree that you are an authorized FlyHigh member prior to login")
                if checkbox_val:
                    st.success("Login successful!")
                    c = c+1

            else:
                st.error("Invalid username/ password")
    if c ==1:
        df_f = flight_df.select_dtypes(include=[np.number])
        cols = flight_df[['Age', 'Gender', 'Customer Type', 'Type of Travel', 'Flight Distance','satisfaction']]
        hiplot_exp = hip.Experiment.from_dataframe(cols)
        hiplot_html = hiplot_exp.to_html()
        st.components.v1.html(hiplot_html, width=800, height=1300)

st.sidebar.markdown(f'<a href="https://flyhighproject-7fmy8ekgwxnxu8d4cwyqhr.streamlit.app"><button>Logout</button> </a>', unsafe_allow_html=True)





        



            
        
        
        
