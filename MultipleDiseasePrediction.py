import pandas as pd
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings("ignore")


# Loading the trained models
diabetes_model = pickle.load(open("C:/Users/User/Downloads/Diabetes_Prediction_Model.sav", 'rb'))
heart_disease_model = pickle.load(open("C:/Users/User/Downloads/Heart_Disease_Model.sav", 'rb'))
breast_cancer_model = pickle.load(open("C:/Users/User/Downloads/Breast_Cancer_Model.sav", 'rb'))
parkinsons_disease_model = pickle.load(open("C:/Users/User/Downloads/Parkinson's_Disease_Model.sav", 'rb'))


# Diabetes prediction function
def diabetes_prediction(age, gender, polyuria, polydipsia, wt_loss, weak, polyphagia, gen_thrush, vis_blur, itch, irritability, del_heal, part_pare, mus_stiff, alop, obes):

    # Converting nominal values to 0 or 1
    binary_map_gen = {"Male": 1, "Female": 0}
    gender = binary_map_gen.get(gender, -1)
    binary_map = {"Yes": 1, "No": 0}
    polyuria = binary_map.get(polyuria, -1)
    polydipsia = binary_map.get(polydipsia, -1)
    wt_loss = binary_map.get(wt_loss, -1)
    weak = binary_map.get(weak, -1)
    polyphagia = binary_map.get(polyphagia, -1)
    gen_thrush = binary_map.get(gen_thrush, -1)
    vis_blur = binary_map.get(vis_blur, -1)
    itch = binary_map.get(itch, -1)
    irritability = binary_map.get(irritability, -1)
    del_heal = binary_map.get(del_heal, -1)
    part_pare = binary_map.get(part_pare, -1)
    mus_stiff = binary_map.get(mus_stiff, -1)
    alop = binary_map.get(alop, -1)
    obes = binary_map.get(obes, -1)

    # Converting the list in a DataFrame
    features_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Polyuria": [polyuria],
        "Polydipsia": [polydipsia],
        "sudden weight loss": [wt_loss],
        "weakness": [weak],
        "Polyphagia": [polyphagia],
        "Genital thrush": [gen_thrush],
        "visual blurring": [vis_blur],
        "Itching": [itch],
        "Irritability": [irritability],
        "delayed healing": [del_heal],
        "partial paresis": [part_pare],
        "muscle stiffness": [mus_stiff],
        "Alopecia": [alop],
        "Obesity": [obes]
    })

    # Predicting diabetes based on the features
    prediction = diabetes_model.predict(features_df)

    # Display the prediction"
    if prediction[0] == 'Positive':
        return "You have Diabetes"
    else:
        return "You do not have Diabetes"



# Heart Disease prediction function
def heartdisease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):

    binary_map_gen = {"Male": 1, "Female": 0}
    sex = binary_map_gen.get(sex, -1)

    # Converting the list in a DataFrame
    features_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalach": [thalach],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal]
    })

    # Predicting heart disease based on the features
    prediction = heart_disease_model.predict(features_df)

    # Display the prediction"
    if prediction[0] == 0:
        return "You do not have Heart Disease"
    else:
        return "You have Heart Disease"



# Breast Cancer prediction function
def breastcancer_prediction(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension):

    # Converting the list in a DataFrame
    features_df = pd.DataFrame({
        "mean_radius": [mean_radius],
        "mean_texture": [mean_texture],
        "mean_perimeter": [mean_perimeter],
        "mean_area": [mean_area],
        "mean_smoothness": [mean_smoothness],
        "mean_compactness": [mean_compactness],
        "mean_concavity": [mean_concavity],
        "mean_concave_points": [mean_concave_points],
        "mean_symmetry": [mean_symmetry],
        "mean_fractal_dimension": [mean_fractal_dimension],
        "radius_error": [radius_error],
        "texture_error": [texture_error],
        "perimeter_error": [perimeter_error],
        "area_error": [area_error],
        "smoothness_error": [smoothness_error],
        "compactness_error": [compactness_error],
        "concavity_error": [concavity_error],
        "concave_points_error": [concave_points_error],
        "symmetry_error": [symmetry_error],
        "fractal_dimension_error": [fractal_dimension_error],
        "worst_radius": [worst_radius],
        "worst_texture": [worst_texture],
        "worst_perimeter": [worst_perimeter],
        "worst_area": [worst_area],
        "worst_smoothness": [worst_smoothness],
        "worst_compactness": [worst_compactness],
        "worst_concavity": [worst_concavity],
        "worst_concave_points": [worst_concave_points],
        "worst_symmetry": [worst_symmetry],
        "worst_fractal_dimension": [worst_fractal_dimension]
    })

    # Predicting heart disease based on the features
    prediction = breast_cancer_model.predict(features_df)

    # Display the prediction"
    if prediction[0] == 0:
        return "You have Malignant Breast Cancer"
    else:
        return "You have Benign Breast Cancer"



# Parkinson's Disease prediction function
def parkinsons_prediction(MDVPFo, MDVPFhi, MDVPFlo, MDVPJitterPercent, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmerDB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE):

    # Converting the list in a DataFrame
    features_df = pd.DataFrame({
        "MDVP:Fo(Hz)": [MDVPFo],
        "MDVP:Fhi(Hz)": [MDVPFhi],
        "MDVP:Flo(Hz)": [MDVPFlo],
        "MDVP:Jitter(%)": [MDVPJitterPercent],
        "MDVP:Jitter(Abs)": [MDVPJitterAbs],
        "MDVP:RAP": [MDVPRAP],
        "MDVP:PPQ": [MDVPPPQ],
        "Jitter:DDP": [JitterDDP],
        "MDVP:Shimmer": [MDVPShimmer],
        "MDVP:Shimmer(dB)": [MDVPShimmerDB],
        "Shimmer:APQ3": [ShimmerAPQ3],
        "Shimmer:APQ5": [ShimmerAPQ5],
        "MDVP:APQ": [MDVPAPQ],
        "Shimmer:DDA": [ShimmerDDA],
        "NHR": [NHR],
        "HNR": [HNR],
        "RPDE": [RPDE],
        "DFA": [DFA],
        "spread1": [spread1],
        "spread2": [spread2],
        "D2": [D2],
        "PPE": [PPE],
    })

    # Predicting parkinson's disease based on the features
    prediction = parkinsons_disease_model.predict(features_df)

    # Display the prediction
    if prediction[0] == 0:
        return "You are Healthy"
    else:
        return "You have Parkinson\'s Disease"



# Adjusting the layout to remove padding and margin
st.set_page_config(layout="wide")


# Footer adjustments
footer_style = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;  /* Adjust padding as needed */
        background-color: #f0f0f0;
        color: #333;
    }
    </style>
    """
st.markdown(footer_style, unsafe_allow_html=True)


# Displaying the sidebar
with st.sidebar:

    style_image1 = """
    width: 60%;
    border-radius: 50% 50% 50% 50%;
    display: block;
    margin-left: auto;
    margin-right: auto;
    """
    
    st.markdown(f'<img src="{"https://w0.peakpx.com/wallpaper/955/713/HD-wallpaper-doctor-medical.jpg"}" style="{style_image1}">', unsafe_allow_html=True,)

    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Breast Cancer Classification',
                            'Parkinson\'s Disease Prediction'],
                            icons=['bandaid', 'activity', 'heart-pulse', 'person'],
                            default_index=0)


# Title container with center-aligned title for all pages
title_container = st.container()
with title_container:
    if selected == 'Diabetes Prediction':
        left_col, mid_col, right_col = st.columns(3)
        with mid_col:
            st.image("https://www.shutterstock.com/image-photo/doctor-checking-blood-sugar-level-600nw-1439349791.jpg", use_column_width=True)
        st.markdown("<h1 style='text-align: center; padding-bottom: 5%;'>Diabetes Prediction</h1>", unsafe_allow_html=True)       
    
    elif selected == 'Heart Disease Prediction':
        left_col, mid_col, right_col = st.columns(3)
        with mid_col:
            st.image("https://www.health365.sg/wp-content/uploads/2022/09/What-is-Ischemic-heart-disease.jpg", use_column_width=True)
        st.markdown("<h1 style='text-align: center; padding-bottom: 5%;'>Heart Disease Prediction</h1>", unsafe_allow_html=True)

    elif selected == 'Breast Cancer Classification':
        left_col, mid_col, right_col = st.columns(3)
        with mid_col:
            st.image("https://t3.ftcdn.net/jpg/06/45/98/46/360_F_645984616_LjQ1RjvNcQl8EXufRXJ3NOJkCrmRZ2OE.jpg", use_column_width=True)
        st.markdown("<h1 style='text-align: center; padding-bottom: 5%;'>Breast Cancer Classification</h1>", unsafe_allow_html=True)

    elif selected == 'Parkinson\'s Disease Prediction':
        left_col, mid_col, right_col = st.columns(3)
        with mid_col:
            st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUbdBScgY5qDeyJS5fcCsdYLHmmOlWcei2RMNQrCi6bVZvus9C35pRMI8NgADR6dyrVy8&usqp=CAU", use_column_width=True)
        st.markdown("<h1 style='text-align: center; padding-bottom: 5%;'>Parkinson's Disease Prediction</h1>", unsafe_allow_html=True)



# Diabetes Prediction page
if (selected == 'Diabetes Prediction'):
    
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input("Enter your Age")
        polyphagia = st.radio("Do you have Polyphagia?", ('Yes', 'No'))
        polyuria = st.radio("Do you have Polyuria?", ('Yes', 'No'))
        polydipsia = st.radio("Do you have polydipsia?", ('Yes', 'No'))
        wt_loss = st.radio("Do you have sudden weight loss?", ('Yes', 'No'))
        weak = st.radio("Do you have Weakness?", ('Yes', 'No'))

    with col2:
        gender = st.selectbox("Enter your Gender", ('Select Gender', 'Male', 'Female'))
        gen_thrush = st.radio("Are you having Genital thrush?", ('Yes', 'No'))
        vis_blur = st.radio("Do you have Visual blurring?", ('Yes', 'No'))
        itch = st.radio("Do you have Itching?", ('Yes', 'No'))
        irritability = st.radio("Do you have Irritability?", ('Yes', 'No'))
    
    with col3:
        del_heal = st.radio("Do you have delayed Healing?", ('Yes', 'No'))
        part_pare = st.radio("Do you have Partial Paresis?", ('Yes', 'No'))
        mus_stiff = st.radio("Do you have Muscle Stiffness?", ('Yes', 'No'))
        alop = st.radio("Do you have Alopecia?", ('Yes', 'No'))
        obes = st.radio("Do you have Obesity?", ('Yes', 'No'))

    # diagnosed message
    diagnosis = ''

    # button
    center_container = st.container()
    with center_container:
        st.markdown("<style>div.stButton > button:first-child { background-color: #b0faff; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton > button:first-child:hover { border-color: blue; color: blue; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton { display: flex; justify-content: center; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton button { width: 20%; }</style>", unsafe_allow_html=True)
        if st.button('Diabetes Test Result', key='diabetes_button'):
            if not age:
                st.error("Please enter Age")
            elif int(age) <= 0:
                st.error("Please enter a valid Age")
            else:
                diagnosis = diabetes_prediction(age, gender, polyuria, polydipsia, wt_loss, weak, polyphagia, gen_thrush, vis_blur, itch, irritability, del_heal, part_pare, mus_stiff, alop, obes)

    # results
    if diagnosis == "You have Diabetes":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #FFB6C1; /* light pink */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    elif diagnosis == "You do not have Diabetes":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #A2FC9F; /* light green */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    # Footer content
    with st.container():
        st.markdown("""<div class="footer">© 2024 Deep Kumar Goenka. All rights reserved.</div>""", unsafe_allow_html=True)



# Heart Disease Prediction page
if (selected == 'Heart Disease Prediction'):

    col1, col2 = st.columns(2)

    with col1:
        age = st.text_input("Enter Age")
        cp = st.text_input("Enter Chest Pain type")
        trestbps = st.text_input("Enter Resting Blood Pressure")
        chol = st.text_input("Enter Serum Cholestoral (mg/dl)")
        fbs = st.text_input("Enter Fasting Blood Sugar (mg/dl)")
        restecg = st.text_input("Enter Resting Electrocardiographic results")
        thal = st.text_input("Enter Thallium Stress test results")

    with col2:
        sex = st.selectbox("Enter your Gender", ('Select Gender', 'Male', 'Female'))
        thalach = st.text_input("Enter maximum Heart Rate achieved")
        exang = st.text_input("Enter exercise induced Angina")
        oldpeak = st.text_input("Enter ST depression induced by exercise relative to rest")
        slope = st.text_input("Enter the slope of the peak exercise ST segment")
        ca = st.text_input("Enter number of major vessels colored by flourosopy")
    

    # diagnosed message
    diagnosis = ''

    # button
    center_container = st.container()
    with center_container:
        st.markdown("<style>div.stButton > button:first-child { background-color: #b0faff; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton > button:first-child:hover { border-color: blue; color: blue; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton { display: flex; justify-content: center; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton button { width: 20%; }</style>", unsafe_allow_html=True)
        if st.button('Heart Disease Test Result', key='heartdisease_button'):
            if not age:
                st.error("Please enter Age")
            elif int(age) <= 0:
                st.error("Please enter a valid Age")
            else:
                diagnosis = heartdisease_prediction(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

    # results
    if diagnosis == "You have Heart Disease":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #FFB6C1; /* light pink */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    elif diagnosis == "You do not have Heart Disease":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #A2FC9F; /* light green */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    # Footer content
    with st.container():
        st.markdown("""<div class="footer">© 2024 Deep Kumar Goenka. All rights reserved.</div>""", unsafe_allow_html=True)



# Breast Cancer Classification page
if (selected == 'Breast Cancer Classification'):

    col1, col2, col3 = st.columns(3)

    with col1:
        mean_radius = st.text_input("Enter Mean Radius")
        mean_texture = st.text_input("Enter Mean Texture")
        mean_perimeter = st.text_input("Enter Mean Perimeter")
        mean_area = st.text_input("Enter Mean Area")
        mean_smoothness = st.text_input("Enter Mean Smoothness")
        mean_compactness = st.text_input("Enter Mean Compactness")
        mean_concavity = st.text_input("Enter Mean Concavity")
        mean_concave_points = st.text_input("Enter Mean Concave Points")
        mean_symmetry = st.text_input("Enter Mean Symmetry")
        mean_fractal_dimension = st.text_input("Enter Mean Fractal Dimension")
    
    with col2:
        radius_error = st.text_input("Enter Radius Error")
        texture_error = st.text_input("Enter Texture Error")
        perimeter_error = st.text_input("Enter Perimeter Error")
        area_error = st.text_input("Enter Area Error")
        smoothness_error = st.text_input("Enter Smoothness Error")
        compactness_error = st.text_input("Enter Compactness Error")
        concavity_error = st.text_input("Enter Concavity Error")
        concave_points_error = st.text_input("Enter Concave Points Error")
        symmetry_error = st.text_input("Enter Symmetry Error")
        fractal_dimension_error = st.text_input("Enter Fractal Dimension Error")
    
    with col3:
        worst_radius = st.text_input("Enter Worst Radius")
        worst_texture = st.text_input("Enter Worst Texture")
        worst_perimeter = st.text_input("Enter Worst Perimeter")
        worst_area = st.text_input("Enter Worst Area")
        worst_smoothness = st.text_input("Enter Worst Smoothness")
        worst_compactness = st.text_input("Enter Worst Compactness")
        worst_concavity = st.text_input("Enter Worst Concavity")
        worst_concave_points = st.text_input("Enter Worst Concave Points")
        worst_symmetry = st.text_input("Enter Worst Symmetry")
        worst_fractal_dimension = st.text_input("Enter Worst Fractal Dimension")
    

    # diagnosed message
    diagnosis = ''

    # button
    center_container = st.container()
    with center_container:
        st.markdown("<style>div.stButton > button:first-child { background-color: #b0faff; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton > button:first-child:hover { border-color: blue; color: blue; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton { display: flex; justify-content: center; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton button { width: 20%; }</style>", unsafe_allow_html=True)
        if st.button('Breast Cancer Test Result', key='breastcancerdisease_button'):
            diagnosis = breastcancer_prediction(mean_radius, mean_texture, mean_perimeter, mean_area, mean_smoothness, mean_compactness, mean_concavity, mean_concave_points, mean_symmetry, mean_fractal_dimension, radius_error, texture_error, perimeter_error, area_error, smoothness_error, compactness_error, concavity_error, concave_points_error, symmetry_error, fractal_dimension_error, worst_radius, worst_texture, worst_perimeter, worst_area, worst_smoothness, worst_compactness, worst_concavity, worst_concave_points, worst_symmetry, worst_fractal_dimension)

    # results
    if diagnosis == "You have Malignant Breast Cancer":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #FFB6C1; /* light pink */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    elif diagnosis == "You have Benign Breast Cancer":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #A2FC9F; /* light green */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    # Footer content
    with st.container():
        st.markdown("""<div class="footer">© 2024 Deep Kumar Goenka. All rights reserved.</div>""", unsafe_allow_html=True)



# Parkinson's Disease Prediction page
if (selected == 'Parkinson\'s Disease Prediction'):

    col1, col2 = st.columns(2)

    with col1:
        MDVPFo = st.text_input('Enter Average vocal fundamental frequency')
        MDVPFhi = st.text_input('Enter Maximum vocal fundamental frequency')
        MDVPFlo = st.text_input('Enter Minimum vocal fundamental frequency')
        MDVPJitterPercent = st.text_input('Enter MDVP:Jitter(%) of fundamental frequency')
        MDVPJitterAbs = st.text_input('Enter MDVP:Jitter(Abs) of fundamental frequency')
        MDVPRAP = st.text_input('Enter MDVP:RAP of fundamental frequency')
        MDVPPPQ = st.text_input('Enter MDVP:PPQ of fundamental frequency')
        JitterDDP = st.text_input('Enter Jitter:DDP of fundamental frequency')
        MDVPShimmer = st.text_input('Enter MDVP:Shimmer of amplitude')
        MDVPShimmerDB = st.text_input('Enter MDVP:Shimmer(dB) of amplitude')
        ShimmerAPQ3 = st.text_input('Enter Shimmer:APQ3 of amplitude')
    
    with col2:
        ShimmerAPQ5 = st.text_input('Enter Shimmer:APQ5 of amplitude')
        MDVPAPQ = st.text_input('Enter MDVP:APQ of amplitude')
        ShimmerDDA = st.text_input('Enter Shimmer:DDA of amplitude')
        NHR = st.text_input('Enter ratio of noise to tonal components (NHR)')
        HNR = st.text_input('Enter ratio of noise to tonal components (HNR)')
        RPDE = st.text_input('Enter nonlinear dynamical complexity measure (RPDE)')
        DFA = st.text_input('Enter Signal fractal scaling exponent')
        spread1 = st.text_input('Enter nonlinear measures of fundamental frequency variation (spread1)')
        spread2 = st.text_input('Enter nonlinear measures of fundamental frequency variation (spread2)')
        D2 = st.text_input('Enter nonlinear dynamical complexity measure (D2)')
        PPE = st.text_input('Enter nonlinear measures of fundamental frequency variation (PPE)')


    # diagnosed message
    diagnosis = ''

    # button
    center_container = st.container()
    with center_container:
        st.markdown("<style>div.stButton > button:first-child { background-color: #b0faff; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton > button:first-child:hover { border-color: blue; color: blue; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton { display: flex; justify-content: center; }</style>", unsafe_allow_html=True)
        st.markdown("<style>div.stButton button { width: 20%; }</style>", unsafe_allow_html=True)
        if st.button('Parkinson\'s Disease Test Result', key='parkinsondisease_button'):
            diagnosis = parkinsons_prediction(MDVPFo, MDVPFhi, MDVPFlo, MDVPJitterPercent, MDVPJitterAbs, MDVPRAP, MDVPPPQ, JitterDDP, MDVPShimmer, MDVPShimmerDB, ShimmerAPQ3, ShimmerAPQ5, MDVPAPQ, ShimmerDDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE)

    # results
    if diagnosis == "You have Parkinson's Disease":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #FFB6C1; /* light pink */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)
    elif diagnosis == "You are Healthy":
        st.markdown("""
        <style>
        .custom-text {
            font-size: 30px;
            text-align: center;
            padding: 10px;
            font-weight: bold;
            font-family: "Times New Roman", Times, serif;
            background-color: #A2FC9F; /* light green */
            border-radius: 15px;
        }
        </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<p class="custom-text">{diagnosis}</p>', unsafe_allow_html=True)

    # Footer content
    with st.container():
        st.markdown("""<div class="footer">© 2024 Deep Kumar Goenka. All rights reserved.</div>""", unsafe_allow_html=True)