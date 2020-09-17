import streamlit as st
import json
import requests
import matplotlib.pyplot as plt
import numpy as np


URI = 'http://127.0.0.1:5000/'

st.title('Neural Network Visualizer')
# st.markdown('<img src = "https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcTYJ04kdH2Xv2m0-X-9bauY4Mjd0o7RDbISCQ&usqp=CAU" style="float:right"> ', unsafe_allow_html=True)
st.markdown('** Ever wondered how a neural network works? **')
st.markdown('** Click on Get random prediction to visualize **')
st.sidebar.markdown('### Neural Network Visualizer')
st.sidebar.markdown('### - Using MNIST data')
st.sidebar.markdown('** @Ankit Sharma **')

st.sidebar.markdown('## Random Input Image')

if st.button('Get random predition'):
    response = requests.post(URI, data={})
    response = json.loads(response.text)
    preds = response.get('prediction')
    image = response.get('image')
    image = np.reshape(image, (28,28))

    st.sidebar.image(image, width = 150)

    for layer, p in enumerate(preds):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize = (32,4))

        if layer == 2:
            row = 1
            col = 10
        else:
            row = 2
            col = 16

        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number*np.ones((8,8,3)))
            plt.xticks([])
            plt.yticks([])

            if layer == 2:
                plt.xlabel(str(i), fontsize = 40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        #activations = ['relu','relu','softmax']
        st.text('Layer {}'.format(layer+1))
        st.pyplot()
            
