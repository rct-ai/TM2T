# coding=utf8
"""================================
@Author: Mr.Chang
@Date  : 2022/12/26 3:07 下午
==================================="""
import streamlit as st
import requests
import json

from streamlit_elements import mui, nivo, elements


def main():

    url = 'http://localhost:8800/t2video'

    headers = {
       'User-Agent': 'apifox/1.0.0 (https://www.apifox.cn)',
       'Content-Type': 'application/json'
    }

    st.title('Hello User!')

    form = st.form(key='my_form')
    text = form.text_input(label='Input a prompt to generate', value='a person is dancing and spinning around')
    submit_button = form.form_submit_button(label='Submit')
    payload = json.dumps({
       "text": text
    })
    if submit_button:

        with st.spinner('Generating...'):
            response = requests.request("POST", url, headers=headers, data=payload, stream=True)
            if response.status_code == 200:
                form.video(response.json()['data'], format='video/mp4')
            else:
                st.write('error: %s' % response.status_code)



if __name__ == '__main__':

    main()
