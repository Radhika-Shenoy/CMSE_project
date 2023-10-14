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
flight_df = pd.read_csv('/Users/radhikavittalshenoy/Downloads/flight_pass_df.csv')
selection = st.sidebar.selectbox("Page View", ["Dashboard", "Analysis Page","Advanced Analysis"])
if selection == "Dashboard":
    
    import streamlit as st

    st.markdown('<h1 style="color:darkblue;font-size:34px;">🛫FlyHigh Airlines: Rise beyond the clouds⛅</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:darkblue;font-size:20px;"><em>We as a brand strive to offer the best experience to our every flyer</em></h3>', unsafe_allow_html=True)

    customer_type_counts = flight_df['satisfaction'].value_counts()

    values = customer_type_counts.values
    labels = customer_type_counts.index

    colors = ['#BFEFFF', '#1E90FE']

    fig = go.Figure(data=go.Pie(values=values, labels=labels, pull=[0.01, 0.04, 0.01, 0.05], hole=0.45, marker_colors=colors))

    fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20)

    fig.add_annotation(x=0.5, y=0.5, text='Satisfaction',
                    font=dict(size=18, family='Verdana', color='black'), showarrow=False)

    fig.update_layout(title_text='Overall passenger satisfaction at FlyHigh Airlines', title_font=dict(size=15, family='Verdana'))

    st.plotly_chart(fig)

    flight_df = pd.read_csv('/Users/radhikavittalshenoy/Downloads/archive-2/passenger_exp_train.csv')
    gender_counts = flight_df['Gender'].value_counts()
    gender_percentage = (gender_counts / len(flight_df)) * 100
    fig = plt.figure(
        FigureClass=Waffle,
        rows=5,
        figsize=(11, 6),
        values=gender_percentage,
        labels=[f"Female ({gender_percentage['Female']:.2f}%)", f"Male ({gender_percentage['Male']:.2f}%)"],  # legend labels with percentages
        colors=["#FF82AB", "#1E90FE"],
        icons=['female', 'male'],
        legend={'loc': 'lower center',
                'bbox_to_anchor': (0.5, -0.5),
                'ncol': len(gender_counts),
                'framealpha': 0.5,
                'fontsize': 14
                },
        icon_size=25,
        icon_legend=True,
        
        title={'label': 'Overall FlyHigh Airlines Gender Distribution pattern',
            'loc': 'left',
            'fontdict': {'fontsize': 17},
            'weight': 'bold'
            }
    )
    st.pyplot(fig)
    
    button_clicked = st.sidebar.button("Satisfaction Confidence")
    if button_clicked:
        fig = px.sunburst(flight_df,path=["Customer Type","satisfaction"],template="plotly")
        fig.update_layout(
            title=dict(
                text="Satisfaction vs Customer Type",
                font=dict(size=15)
            )
        )

        st.plotly_chart(fig)
    
    age_button_clicked = st.sidebar.button("Age stats")
    if age_button_clicked:

        def categorize_age(age):
            if age <= 1:
                return 'Newborns'
            elif 1 < age <= 3:
                return 'Infants'
            elif 3 < age <= 18:
                return 'Children'
            elif 18 < age <= 60:
                return 'Adults'
            else:
                return 'Seniors'
        column1,column2 = st.columns([1,1])
        with column1:
            flight_df['Age_Group'] = flight_df['Age'].apply(categorize_age)
            age_counts = flight_df['Age_Group'].value_counts()

            colors = ['#1E90FF', '#98F5FF']

            fig = px.pie(
                values=age_counts.values,
                names=age_counts.index,
                title="Traveller Age category distribution",
                color_discrete_sequence=colors 
            )

            fig.update_traces(textinfo='percent+label', pull=[0.03, 0.02], textfont=dict(size=18)) 

            fig.update_layout(
                showlegend=True,
                title_font=dict(size=15),
                width=350,
                height=600
            )
            st.plotly_chart(fig)

        with column2:
            age_count = flight_df['Age'].value_counts().reset_index()
            fig = px.bar(
                age_count,
                x='Age',
                y='count',
                title='Age Group Analysis',
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
                title_font=dict(size=16),
                showlegend=False,
                plot_bgcolor='#FFFFFF',
                margin=dict(l=25, r=20, t=100, b=30),
            )

            st.plotly_chart(fig)

elif selection == 'Analysis Page':
    selected_features = []
    flight_df = flight_df.drop(columns=['Unnamed: 0', 'id'])
    columns = list(flight_df.columns)
    categorical_data = list(set(flight_df.columns) - set(flight_df._get_numeric_data().columns))
    non_categorical_data = list(set(columns) - set(categorical_data))   
    feature_options = st.multiselect('Select 2 Features', non_categorical_data, key="multiselect") 
    
    if len(feature_options)!=2:
        st.warning('Please select 2 features for viewing graphical visualization')
    if len(feature_options) >0:
        plot_options = st.selectbox('Plot options', ('None','Scatter plot', 'Histogram', 'Pie chart'))
        if plot_options == 'Scatter plot':
            
            categorical_feature = st.selectbox('Select the feature against which you want to analyse the already selected feature', categorical_data)
            st.pyplot(sns.scatterplot(data=flight_df, x=feature_options[0], y=feature_options[1], hue=categorical_feature).figure)
                
        elif plot_options == 'Histogram':
            fig=px.histogram(flight_df,x=feature_options[0],color=flight_df[random.choice(categorical_data)],title=f"{feature_options[0]} vs {feature_options[1]}",
                    color_discrete_sequence=px.colors.qualitative.Vivid)
            fig.update_layout(template="plotly")
            fig.update_layout(title_font_size=30)
            st.plotly_chart(fig)

elif selection == 'Advanced Analysis':
    
        def main():
            username = st.text_input("Enter your FlyHigh admin userID:")
            password = st.text_input("Enter your FlyHigh admin password:", type="password")
            checkbox_val = st.checkbox("I am an authorized FlyHigh member from the R&D team and wish to view the high dimensional representation of features!")
            if st.button("Login"):
                if username == 'user' and password == 'password':               
                    if not checkbox_val:
                        st.write("Please agree that you are an authorized FlyHigh member prior to login")
                    if checkbox_val:
                        st.write("Login successful!")
                        df_f = flight_df.select_dtypes(include=[np.number])
                        cols = flight_df[['Age', 'Gender', 'Customer Type', 'Type of Travel', 'Flight Distance']]
                        hiplot_exp = hip.Experiment.from_dataframe(cols)
                        hiplot_html = hiplot_exp.to_html()
                        st.components.v1.html(hiplot_html, width=700, height=1500)
                else:
                    st.write("Invalid username/ password")

        if __name__ == '__main__':
            main()



            
        
        
        
