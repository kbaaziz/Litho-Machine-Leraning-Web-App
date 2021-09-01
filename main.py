from operator import index
from pandas.core.arrays import categorical
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
from tensorflow import keras
import os 
from utilities import fixed_clustering_plot, litho_confusion_matrix, scatter_plot, predictions_plot
st.set_option('deprecation.showPyplotGlobalUse', False)


def _max_width_():
    max_width_str = f"max-width: 1200px;"
    st.markdown(
        f"""
    <style>
    .reportview-container .main .block-container{{
        {max_width_str}
    }}
    </style>    
    """,
        unsafe_allow_html=True,
    )

_max_width_()


st.title("MACHINE AND DEEP LEARNING FOR LITHOLOGY PREDICTION IN THE NORTH SEA - FORCE DATASET")
st.markdown("<h3 style='text-align: right; color: black;'> Author: John Masapanta Pozo</h3>", unsafe_allow_html=True)
st.sidebar.title("VISUALIZATION SELECTION")
st.sidebar.markdown("""<img src="https://allaboutintelligence.files.wordpress.com/2020/09/deep-learning-methodologies-and-applications.gif?w=775" style="width: 100%; cursor: default;" >""", unsafe_allow_html=True)
#st.sidebar.markdown("Select the well you want to visualize:")


#--------------------------------------

## 1. SELECTING DATASET

st.sidebar.markdown("-------")
st.sidebar.subheader("SELECTING DATASET")
selected_dataset = st.sidebar.radio("Select the dataset you want to predict and visualize:", options=['Open dataset', 'Hidden dataset'], index=1)

#loading raw data
@st.cache
def load_raw():

    #main_directory = 'C:/Users/Alexandra/Desktop/GEOSCIENCES MASTERS PROGRAM/THESIS PROJECT/Thesis/App_thesis/raw_data/'
    
    lithology_numbers = {30000: 0, 65030: 1, 65000: 2, 80000: 3, 74000: 4, 70000: 5, 
                        70032: 6, 88000: 7, 86000: 8, 99000: 9, 90000: 10, 93000: 11
                        }

    #raw_hidden = pd.read_csv(main_directory + "\hidden_test.csv", sep=";")
    raw_hidden = pd.read_csv(r"./raw_data/hidden_test.csv", sep=";")

    raw_hidden = raw_hidden.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
    raw_hidden = raw_hidden.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
    raw_hidden['LITHO'] = raw_hidden["LITHO"].map(lithology_numbers)

    test_data = pd.read_csv("./raw_data/open_test.csv", sep=';')
    test_labels = pd.read_csv(main_directory + "./raw_data/open_target.csv", sep=';')
    raw_open = pd.merge(test_data, test_labels, on=['WELL', 'DEPTH_MD']) 
    raw_open = raw_open.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
    raw_open['LITHO'] = raw_open["LITHO"].map(lithology_numbers)

    return raw_open, raw_hidden

raw_open, raw_hidden = load_raw()


#loading treated data
#@st.cache
def load_treated(file_name):

    if os.path.exists(os.path.join(r"./real_time_predictions", file_name + '.csv')):
        loaded_df = pd.read_csv(r'./real_time_predictions/'+ file_name + '.csv')
    else:
        #df_hidden_treated.to_csv(os.path.join(r"./real_time_predictions/" , file_name + '.csv'))
        loaded_df = pd.read_csv(r'./treated_data/'+ file_name + '.csv')

    return loaded_df

data_dict = {'Open dataset': 'open_data', 'Hidden dataset': 'hidden_data'}


if selected_dataset == 'Open dataset':
    raw_data = raw_open
    # file_name = data_dict[selected_dataset]
    # selected_data = load_treated(file_name)
else:
    raw_data = raw_hidden
    # file_name = data_dict[selected_dataset]
    # selected_data = load_treated(file_name)

file_name = data_dict[selected_dataset]
selected_data = load_treated(file_name)



#---------------------------------

#2. RAW DATA VISUALIZATION

st.sidebar.markdown("-------")
st.sidebar.subheader("RAW DATA VISUALIZATION")

#selecting well
raw_well = st.sidebar.selectbox("Select the well to be visualized", raw_data.WELL.unique(), index=0)
raw_well_df = raw_data[raw_data.WELL == raw_well].set_index("DEPTH_MD")

#selecting logs
raw_logs = raw_well_df.columns.drop(["WELL", "X_LOC", "Y_LOC", "Z_LOC", "GROUP",
                                    "FORMATION", 'LITHO'])
raw_continuous_logs = st.sidebar.multiselect("Select the well logs to display", raw_logs, default='GR', key='raw_logs')
raw_facies_logs = ['LITHO']
raw_cols = [*raw_continuous_logs, *raw_facies_logs]

raw_well_df = raw_well_df[raw_cols]

if st.sidebar.button('Plot raw logs', key='raw_logs_plot'):
#if st.sidebar.checkbox("PLOT PREDICTIONS", True, key='1'):
    st.markdown("### RAW LOGS AND ACTUAL FACIES")
    st.markdown("<h3 style='text-align: center; color: black;'> WELL {} </h3>".format(raw_well), unsafe_allow_html=True)
    pred_plot = fixed_clustering_plot(raw_well_df, raw_cols, raw_facies_logs)
    st.pyplot(fig=pred_plot, use_container_width=True)


#---------------------------------------



## 2. MAKING PREDICTIONS REAL TIME

st.sidebar.markdown("-------")
st.sidebar.subheader("REAL TIME PREDICTIONS")

# uploading pretrained models
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost
import lightgbm
import catboost

#@st.cache
def pretrained_models():
    desicion_tree = pickle.load(open(r"./models/decision_tree.pkl", 'rb'))
    k_nn = pickle.load(open(r"./models/k_nearest_neight.pkl", 'rb'))
    logistic_regression = pickle.load(open(r"./models/logistic_regression.pkl", 'rb'))
    #extreme_gb = pickle.load(open('models\extreme_gb.sav', 'rb'))
    categorical_gb = pickle.load(open(r"./models/categorical_gb.pkl", 'rb'))
    light_gb = pickle.load(open(r"./models/light_gb.pkl", 'rb'))
    neural_network = keras.models.load_model(r"./models/neural_network.h5")
    
    return desicion_tree, k_nn, logistic_regression, light_gb, categorical_gb, neural_network

dt_model, knn_model, lr_model, light_model, cat_model, nn_model = pretrained_models()



models = [lr_model, dt_model, knn_model, nn_model, light_model, cat_model]
models_names = ['Logistic Regression', 'Decision Tree', 'K-NN', 'Neural Network', 'LGBMBOOST', 'CATBOOST']
models_dict = dict(zip(models_names, models))

def features_selection(model_name):
    
    if model_name == 'Logistic Regression':
        selected_features = ['DTS_COMB', 'G', 'P_I', 'GR','NPHI_COMB', 
                                'DTC', 'RHOB', 'DT_R', 'Z_LOC', 'S_I','K']
    
    elif model_name == 'Decision Tree':
        selected_features = ['Cluster', 'DEPTH_MD', 'X_LOC', 'Y_LOC', 'Z_LOC', 'CALI',
                            'RSHA', 'RMED', 'RDEP', 'RHOB', 'GR', 'NPHI', 'PEF', 'DTC', 'SP', 'BS',
                            'ROP', 'DTS', 'DCAL', 'DRHO', 'RMIC', 'GROUP_encoded',
                            'FORMATION_encoded', 'WELL_encoded', 'DTS_pred', 'DTS_COMB',
                            'NPHI_pred', 'NPHI_COMB', 'RHOB_pred', 'RHOB_COMB', 'DTC_pred',
                            'DTC_COMB', 'S_I', 'P_I', 'DT_R', 'G', 'K', 'MD_TVD']

    elif model_name == 'K-NN':
        selected_features = ['GR', 'FORMATION_encoded', 'GROUP_encoded', 'NPHI_COMB', 'RHOB', 
                            'X_LOC', 'BS', 'CALI', 'SP', 'WELL_encoded', 'Z_LOC', 'DT_R', 'DEPTH_MD', 'DTC', 'Cluster']
    
    elif model_name == 'Neural Network':
        selected_features = ['GROUP_encoded', 'GR', 'NPHI_COMB', 'Y_LOC', 'RHOB',
                            'DEPTH_MD', 'FORMATION_encoded', 'Z_LOC', 'WELL_encoded', 'X_LOC',
                            'RMED', 'CALI', 'DTC', 'MD_TVD', 'DT_R',
                            'PEF', 'RDEP', 'DTS_COMB', 'G', 'SP',
                            'Cluster', 'K', 'P_I', 'DRHO', 'DCAL']

    elif model_name == 'LGBMBOOST':
        selected_features = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'DTC', 'DTS_COMB', 'RSHA', 'DT_R',
                            'RHOB', 'K', 'DCAL', 'Y_LOC', 'GROUP_encoded', 'WELL_encoded',
                            'DEPTH_MD', 'Z_LOC', 'CALI', 'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD',
                            'ROP', 'DRHO']

    elif model_name == 'CATBOOST':
        selected_features = ['GR', 'NPHI_COMB', 'DTC', 'DTS_COMB','RHOB',
                            'Y_LOC', 'GROUP_encoded', 'WELL_encoded', 
                            'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI',
                            'X_LOC', 'RMED', 'SP', 'MD_TVD']
    
    return selected_features 

#Model selection
selected_models = st.sidebar.multiselect("Select model(s) to use during the predictions:", ['Logistic Regression', 'Decision Tree', 'K-NN', 'LGBMBOOST', 'CATBOOST', 'Neural Network'], default='LGBMBOOST')

models_header = {'Logistic Regression': 'LR_PRED',
                'Decision Tree': 'DT_PRED', 
                'K-NN': 'KNN_PRED',
                'LGBMBOOST': 'LGBM_PRED',
                'CATBOOST': 'CAT_PRED',
                'Neural Network' : 'NN_PRED'
                }

#Predictions
results_data = raw_data.copy()

if st.sidebar.checkbox('Store results in cache:', True, key='updating_predictions'):

    if st.sidebar.button('RUN', key='predict'):

        for model_name in selected_models:

            if model_name != 'Neural Network':

                if not models_header[model_name] in selected_data.columns:

                    model_i = models_dict[model_name]
                    selected_features = features_selection(model_name)

                    y_hat = model_i.predict(selected_data[selected_features])
                    selected_data[models_header[model_name]] = y_hat

                    #storing results
                    selected_data.to_csv(r'./real_time_predictions/'+ file_name + '.csv')

                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]       #results frame

                else:
                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]   #results frame
                
                #results_data.to_csv(os.path.join(r"./predictions_visual/" , file_name + '.csv'))
            
            else:

                if not models_header[model_name] in selected_data.columns:

                    model_i = models_dict[model_name]
                    selected_features = features_selection(model_name)

                    y_hat_prob = model_i.predict(selected_data[selected_features])
                    y_hat = np.array(pd.DataFrame(y_hat_prob).idxmax(axis=1))
                    selected_data[models_header[model_name]] = y_hat

                    #storing results
                    selected_data.to_csv(r'./real_time_predictions/'+ file_name + '.csv')

                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]                  #results frame
                    #selected_data.to_csv(r'./real_time_predictions/'+ file_name + '.csv')
                
                else:
                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]    #results frame
                    #results_data.to_csv(r'./real_time_predictions/'+ 'hidden_res_saving' + '.csv')

            results_data.to_csv(os.path.join(r"./predictions_visual/" , file_name + '.csv'))
else:

    if st.sidebar.button('RUN', key='predict'):

        for model_name in selected_models:

            if model_name != 'Neural Network':

                if not models_header[model_name] in selected_data.columns:

                    model_i = models_dict[model_name]
                    selected_features = features_selection(model_name)

                    y_hat = model_i.predict(selected_data[selected_features])
                    selected_data[models_header[model_name]] = y_hat

                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]                  #results frame
                    #selected_data.to_csv(r'./real_time_predictions/'+ file_name + '.csv')
                
                else:
                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]    #results frame
                    #results_data.to_csv(r'./real_time_predictions/'+ 'hidden_res_saving' + '.csv')

                #results_data.to_csv(os.path.join(r"./predictions_visual/" , file_name + '.csv'))
            
            else:

                if not models_header[model_name] in selected_data.columns:

                    model_i = models_dict[model_name]
                    selected_features = features_selection(model_name)
                    
                    y_hat_prob = model_i.predict(selected_data[selected_features])
                    y_hat = np.array(pd.DataFrame(y_hat_prob).idxmax(axis=1))

                    selected_data[models_header[model_name]] = y_hat

                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]                  #results frame
                    #selected_data.to_csv(r'./real_time_predictions/'+ file_name + '.csv')
                
                else:
                    results_data[models_header[model_name]] = selected_data[models_header[model_name]]    #results frame
                    #results_data.to_csv(r'./real_time_predictions/'+ 'hidden_res_saving' + '.csv')

            results_data.to_csv(os.path.join(r"./predictions_visual/" , file_name + '.csv'))


st.sidebar.markdown('----------------')
st.sidebar.subheader("RESULTS VISUALIZATION")

import sklearn
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report



#if st.sidebar.checkbox('Show predition results:', False, key='showing_predictions'):

if not os.path.exists(os.path.join(r"./predictions_visual/" , file_name + '.csv')):

    st.text('NO STORED RESULTS IN LOCAL DISK. PLEASE RUN PREDICTIONS SECTIONS FIRST...')

else:

    results_data = pd.read_csv(os.path.join(r"./predictions_visual/" , file_name + '.csv'))

    if all(elem in results_data.columns for elem in [models_header[model_name] for model_name in  selected_models]):

        selected_well = st.sidebar.selectbox("Select the well to be visualized", results_data.WELL.unique(), index=0, key='predicted_well')
        selected_well_df = results_data[results_data.WELL == selected_well].set_index("DEPTH_MD")

        facies_logs_names = [models_header[model_name] for model_name in selected_models if models_header[model_name] in results_data.columns] + ['LITHO']

        logs_drop = ["WELL", "X_LOC", "Y_LOC", "Z_LOC", "GROUP", "FORMATION", 'LITHO'] + facies_logs_names
        log_names = selected_well_df.columns.drop(logs_drop)

        # continuous_logs = st.sidebar.multiselect("Select the well logs to display:", log_names, default='GR', key='no_raw_logs')
        # facies_logs = st.sidebar.multiselect("Select facies logs to visualize:", facies_logs_names, default='LITHO')


        with st.container():

            #selecting logs
            continuous_logs = st.sidebar.multiselect("Select the well logs to display:", log_names, default='GR', key='no_raw_logs')
            facies_logs = st.sidebar.multiselect("Select facies logs to visualize:", facies_logs_names, default='LITHO')

            #final dataframe
            cols = [*continuous_logs, *facies_logs]
            DF = selected_well_df[cols]

            #Showing scatter plots
            if st.sidebar.checkbox('Show scatter plots and classification reports:', False, key='showing_scatter'):

                col1, col2 = st.columns(2)
                with col1:
                    model1 = st.sidebar.selectbox('Model 1 to display:', facies_logs, index=0, key='model1')
                    scat1 = st.sidebar.multiselect('First scatter plot:', log_names, default=['GR', 'RHOB'])
                    #color1 = st.sidebar.selectbox('Color coded by:', facies_logs, index=0, key='color_label1')
                with col2:
                    model2 = st.sidebar.selectbox('Second scatter plot:', facies_logs, index=0, key='model2')
                    scat2 = st.sidebar.multiselect('Model 2 to display:', log_names, default=['NPHI', 'DTC'])

                if st.sidebar.button('Plot predictions', key='predictions'):
                    # if st.sidebar.checkbox('Show scatter plots', True, key='2'):
                    st.markdown("## **PREDICTED FACIES**")
                    st.markdown("<h3 style='text-align: center; color: black;'> WELL {} </h3>".format(selected_well), unsafe_allow_html=True)
                    pred_plot = predictions_plot(DF, cols, facies_logs)
                    st.pyplot(fig=pred_plot, use_container_width=True)

                    
                    st.markdown('------------')
                    st.markdown("## **STATISTICS VISUALIZATION**")
                    col1, col2 = st.columns(2)
                    with col1:
                        #color1 = st.sidebar.selectbox('Color coded by:', facies_logs, index=0, key='color_label1')
                        st.markdown("<h3 style='text-align: center; color: black;'> {} PREDICTED LITHOLOGY</h3>".format(model1), unsafe_allow_html=True)
                        st.plotly_chart(scatter_plot(selected_well_df, scat1[0], scat1[1], model1))
                        st.plotly_chart(scatter_plot(selected_well_df, scat2[0], scat2[1], model1))

                    with col2:
                        st.markdown("<h3 style='text-align: center; color: black;'> {} PREDICTED LITHOLOGY</h3>".format(model2), unsafe_allow_html=True)
                        st.plotly_chart(scatter_plot(selected_well_df, scat1[0], scat1[1], model2))
                        st.plotly_chart(scatter_plot(selected_well_df, scat2[0], scat2[1], model2))

                    # conf_matrix = litho_confusion_matrix(DF[actual_litho], DF[model_litho])
                    # st.pyplot(fig=conf_matrix)

                                    
                    st.markdown('------------')
                    st.markdown("## **CLASSIFICATION REPORTS**")
                    
                    #showing classification reports
                    # cols_names = ['col'+ str(i) for i in range(len(selected_models))]
                    # cols_names = st.columns(len(cols_names))

                    col_1, col_2 = st.columns(2)
                    with col_1:
                        st.markdown("<h3 style='text-align: center; color: black;'> {} CLASSIFICATION REPORT</h3>".format(model1), unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(classification_report(results_data.LITHO, results_data[model1], output_dict=True)).T)

                    with col_2:
                        st.markdown("<h3 style='text-align: center; color: black;'> {} CLASSIFICATION REPORT</h3>".format(model2), unsafe_allow_html=True)
                        st.dataframe(pd.DataFrame(classification_report(results_data.LITHO, results_data[model2], output_dict=True)).T)

            else:

                if st.sidebar.button('Plot predictions', key='predictions'):

                    st.markdown("### PREDICTED FACIES")
                    st.markdown("<h3 style='text-align: center; color: black;'> WELL {} </h3>".format(selected_well), unsafe_allow_html=True)
                    pred_plot = predictions_plot(DF, cols, facies_logs)
                    st.pyplot(fig=pred_plot, use_container_width=True)


    else:
        st.text('SOME MODELS YOU WANT TO VISUALIZE ARE MISSING IN THE STORED RESULTS. PLEASE RUN PREDICTIONS SECTIONS FIRST...')
