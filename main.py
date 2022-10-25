import streamlit as st
import numpy as np
import pandas as pd
import wave
import matplotlib.pyplot as plt
import pickle

# loading the saved model
model_SVM = pickle.load(open("model_SVM.pkl",'rb'))
model_NB = pickle.load(open("model_NB.pkl",'rb'))
MMS = pickle.load(open("minmaxscaler.pkl",'rb'))


st.markdown("<h1 style='text-align: center; color: white; font-size: 50px;'>AI-SKEMA</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: white; font-size: 30px;'>Aplikasi Identifikasi Suara Kemacetan</h1>", unsafe_allow_html=True)


uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
        audio_bytes = wave.open(uploaded_file)
        signal_sf = audio_bytes.readframes(-1)
        st.audio(uploaded_file, format='audio/wav')

        # Convert audio bytes to integers
        soundwave_sf = np.frombuffer(signal_sf, dtype='int16')

        # Get the sound wave frame rate
        framerate_sf = audio_bytes.getframerate()*2
        print(framerate_sf)

        # Find the sound wave timestamps
        time_sf = np.linspace(start=0,
                            stop=len(soundwave_sf)/framerate_sf,
                            num=len(soundwave_sf))

        #GET AUDIO DATA AND DISPLAY IN GRAPH
        fig, ax = plt.subplots(figsize=(15, 3))
        plt.plot(time_sf, soundwave_sf)
        plt.title('Amplitude over Time')
        plt.ylabel('Amplitude')
        plt.xlabel('Time (seconds)')

        with st.container():
            st.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Grafik Suara yang di Upload</h1>", unsafe_allow_html=True)
            st.pyplot(fig)


        #GET AUDIO FEATURE AND DISPLAY IN DATAFRAME
        percentile = np.percentile(np.array(soundwave_sf),[25,50,75])

        list_Q1 = []
        list_Q2 = []
        list_Q3 = []
        list_MAX = []
        list_MIN = []
        list_Mean = []
        list_Median = []
        list_StandardDeviation = []
        list_Variance = []

        # list_Q1.append(percentile[0])
        # list_Q2.append(percentile[1])
        list_Q3.append(percentile[2])
        # list_MAX.append(np.amax(np.array(soundwave_sf)))
        # list_MIN.append(np.amin(np.array(soundwave_sf)))
        list_Mean.append(np.mean(np.array(soundwave_sf)))
        list_Median.append(np.median(np.array(soundwave_sf)))
        list_StandardDeviation.append(np.std(np.array(soundwave_sf)))
        list_Variance.append(np.var(np.array(soundwave_sf)))

        dataframe = {'Q3': list_Q3, 'Mean':list_Mean, 'Median':list_Median, 'StdDeviation': list_StandardDeviation, 'Variance': list_Variance}
        df = pd.DataFrame(dataframe)

        col1, col2, col3 = st.columns([2,8,2])

        col1.subheader("")
        col1.write("")

        col2.markdown("<h1 style='text-align: center; color: white; font-size: 20px;'>Dataframe Fitur</h1>", unsafe_allow_html=True)
        col2.dataframe(df)

        col3.subheader("")
        col3.write("")
        

        def main(df):
            df = MMS.transform(df)
            st.markdown("----", unsafe_allow_html=True)
            model_select = st.selectbox(
            'Pilih Algoritma',
            ('Support Vector Machine', 'Naive Bayes'))
            st.write('Anda Memilih:', model_select)
            if model_select == 'Support Vector Machine':
                model = model_SVM.predict(df)
                df_proba = model_SVM.predict_proba(df)
            elif model_select == "Naive Bayes":
                model = model_NB.predict(df)
                df_proba = model_NB.predict_proba(df)
            
            

            columns1 = st.columns((2, 2, 2))
            if columns1[1].button('Hasil Identifikasi Audio'):
                if model == 1:
                    st.success('Audio ter-identifikasi TIDAK MACET dengan probabilitas '+"{:.2f}".format(df_proba.tolist()[0][1]*100)+'%')
                else:
                    st.error('Audio ter-identifikasi MACET dengan probabilitas '+"{:.2f}".format(df_proba.tolist()[0][0]*100)+'%')
                
            st.markdown("----", unsafe_allow_html=True)
            
                

        if __name__ == '__main__':
                main(df)