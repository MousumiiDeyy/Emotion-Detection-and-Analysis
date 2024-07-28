import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import joblib

pipe_lr = joblib.load(open("emotion_classifier_pipe_lr_23_march_2024.pkl","rb"))

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}

def main():
  st.title("Emotion classifier")
  menu=["home","monitor","About"]
  choice = st.sidebar.selectbox("Menu",menu)

  if choice == "home":
    st.subheader("Home-Emotion In Text")

    with st.form(key='emotion_clf_form'):
      raw_text = st.text_area("Type here")
      submit_text = st.form_submit_button(label='Submit')
    if submit_text:
      col1,col2 = st.columns(2)

      prediction = predict_emotions(raw_text)
      probability = get_prediction_proba(raw_text)

      with col1:
        st.success("Original Text")
        st.write(raw_text)

        st.success("Prediction")
        emoji_icon = emotions_emoji_dict[prediction]
        st.write("{}:{}".format(prediction,emoji_icon))
        st.write("Confidence:{}".format(np.max(probability)))

      with col2:
        st.success("Prediction Probability")
        st.write(probability)
        proba_df = pd.DataFrame(probability, columns=pipe_lr.classes_)
        st.write(proba_df.T)
        proba_df_clean =  proba_df.T.reset_index()
        proba_df_clean.columns = ["emotions","probability"]

        fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability')
        st.altair_chart(fig,use_container_width=True)


  elif choice =="Monitor":
    st.subheader("Monitor app")
  else:
    st.subheader("about")

if __name__ =='__main__':
  main()