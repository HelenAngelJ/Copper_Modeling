import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title= "Copper Modeling",
                   layout= "wide")
title = '<div style="text-align: center;font-size: 58px; color:#323232;">Industrial Copper Modeling</div>'
st.markdown(title, unsafe_allow_html=True)

tab1,tab2=st.tabs(['Price Prediction','Status Prediction'])

#definind the unique values for categorical attributes
item_type = ['W', 'S', 'PL', 'Others', 'WI', 'IPL', 'SLAWR']
status = ['Won', 'Lost', 'Not lost for AM', 'Revised', 'To be approved', 'Draft',
        'Offered', 'Offerable', 'Wonderful']
application = [10., 41., 28., 59., 15.,  4., 38., 56., 42., 26., 27., 19., 20.,
        66., 29., 22., 40., 25., 67., 79.,  3., 99.,  2.,  5., 39., 69.,
        70., 65., 58., 68.]
country = [ 28.,  25.,  30.,  32.,  38.,  78.,  27.,  77., 113.,  79.,  26.,
            39.,  40.,  84.,  80., 107.,  89.]
product_reference = [1670798778, 1668701718,     628377,     640665,     611993,
        1668701376,  164141591, 1671863738, 1332077137,     640405,
        1693867550, 1665572374, 1282007633, 1668701698,     628117,
        1690738206,     628112,     640400, 1671876026,  164336407,
            164337175, 1668701725, 1665572032,     611728, 1721130331,
        1693867563,     611733, 1690738219, 1722207579,  929423819,
        1665584320, 1665584662, 1665584642]

with tab1:
    with st.form("my_form",clear_on_submit=True):
        col1,col3,col2 = st.columns([0.4,0.2,0.4])
        with col1:
            quantity_tons = st.number_input('Quantity tons (min:0.00001,max:1000000000.0)',min_value=0.00001,max_value=1000000000.0,format='%0.5f')
            thickess = st.number_input('Thickness (min:0.180,max:400.0)',min_value=0.180,max_value=400.0)
            width = st.number_input('width (min:1.0,max:2990.0)',min_value=1.0,max_value=2990.0)
            Status = st.selectbox('Status',status)
            customer = st.number_input("Customer ID")
            
        with col2:
            Item_type = st.selectbox('Item Type',item_type)
            appli_cation = st.selectbox('Application',application)
            Country = st.selectbox('Country',country)
            pro_duct_ref = st.selectbox('Product Reference',product_reference)
            predict_submit = st.form_submit_button("predict Selling price")

        if predict_submit:
            with open(r"choosen_model_decTreeReg.pkl", 'rb') as model:
                predict_model = pickle.load(model)
            with open(r'std_scaler.pkl', 'rb') as std:
                scaler_loaded = pickle.load(std)
            with open(r"item.pkl", 'rb') as item:
                item_loaded = pickle.load(item)
            with open(r"status.pkl", 'rb') as stat:
                status_loaded = pickle.load(stat)
            
            predict_sample = np.array([[np.log(float(quantity_tons)),Status,Item_type, appli_cation, np.log(float(thickess)),float(width), Country, int(customer), int(pro_duct_ref)]])
            sample_OEI = item_loaded.transform(predict_sample[:, [2]])
            sample_OES = status_loaded.transform(predict_sample[:, [1]])
            sample = np.concatenate(
                (predict_sample[:, [0]], sample_OES,sample_OEI,predict_sample[:,[3,4,5,6,7,8]]),
                axis=1)
            predict_input = scaler_loaded.transform(sample)
            prediction = predict_model.predict(predict_input)[0]

            
            st.write('## :green[Predicted selling price:] ', np.exp(prediction))


with tab2:
    with st.form("my_form1",clear_on_submit=True):
        col1,col3,col2 = st.columns([0.4,0.2,0.4])
        with col1:
            cQuantity_tons = st.number_input('Quantity tons (min:0.00001,max:1000000000.0)',min_value=0.00001,max_value=1000000000.0,format='%0.5f')
            cThickness = st.number_input('Thickness (min:0.180,max:400.0)',min_value=0.180,max_value=400.0)
            cWidth = st.number_input('width (min:1.0,max:2990.0)',min_value=1.0,max_value=2990.0)
            cSelling_price = st.number_input('selling price (min:0.1,max:2990.0)',min_value=0.1,max_value=100001015.0)
            cCustomer = st.number_input('Customer ID')
                        
        with col2:
            cItem_type = st.selectbox('Item Type',item_type)
            cApplication = st.selectbox('Application',application)
            cCountry = st.selectbox('Country',country)
            cProduct_ref = st.selectbox('Product Reference',product_reference)
            class_submit = st.form_submit_button("predict Status")

        if class_submit:
            with open(r"ExtraTreesClassifier_model.pkl", 'rb') as file:
                cloaded_model = pickle.load(file)
            with open(r'cscaler.pkl', 'rb') as f:
                cscaler_loaded = pickle.load(f)
            with open(r"ct.pkl", 'rb') as f:
                ct_loaded = pickle.load(f)

            predict_class = np.array(
                                [[np.log(float(cQuantity_tons)), np.log(float(cSelling_price)),cItem_type,cApplication,
                                  np.log(float(cThickness)), float(cWidth), cCountry, int(cCustomer),
                                  int(cProduct_ref)]])
            OE_class_item = ct_loaded.transform(predict_class[:, [2]])
            new_sample = np.concatenate((predict_class[:, [0, 1]], OE_class_item,predict_class[:,[3,4,5,6,7,8]]),
                                        axis=1)
            pred_class_input = cscaler_loaded.transform(new_sample)
            pred_class = cloaded_model.predict(pred_class_input)
            #st.write(pred_class)
            if pred_class == 1:
                st.write('## :green[The Status is Won] ')
            else:
                st.write('## :red[The status is Lost] ')
            

