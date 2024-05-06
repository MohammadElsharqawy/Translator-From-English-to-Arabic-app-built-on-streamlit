from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import streamlit as st

num_beams = 5

tokenizer = AutoTokenizer.from_pretrained("marefa-nlp/marefa-mt-en-ar")
model = AutoModelForSeq2SeqLM.from_pretrained("marefa-nlp/marefa-mt-en-ar")


st.set_page_config(page_title='Simply! Translate ', layout='wide', initial_sidebar_state='expanded')

st.title(" Translator From English to Arabic:balloon:")
text = st.text_area("Enter text:",height=None,max_chars=None,key=None,help="Enter your text here")

if st.button('Translate Sentence'):
    if text == "":
        st.warning('Please **enter text** for translation')
    else:
        input_ids4 = tokenizer(text, return_tensors="pt").input_ids
        outputs = model.generate(input_ids4, max_length=50, num_beams=num_beams, early_stopping=True,)
        st.info(str(tokenizer.decode(outputs[0], skip_special_tokens=True)))
        st.success("Translation is **successfully** completed!")
        st.balloons()
else:
  pass        

