#!/usr/bin/env python
# coding: utf-8

# In[43]:


from flask import Flask


# In[44]:


app = Flask(__name__)


# In[45]:


import re # Regular Expression library (Regular expression is a technique for text parsing/manipulation)
import nltk # Natural Language Tool Kit library (NLTK)
nltk.download('stopwords') # Download the library of stopwords (Words that have no value/meaning such as the/as/do/is)
from nltk.corpus import stopwords # Import the stopwords library
from nltk.stem.porter import PorterStemmer


# In[46]:


from flask import request, render_template
import joblib 

@app.route("/", methods=["GET", "POST"]) #this line is decorator: confirm must put de
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower()
        text = text.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        text = [ps.stem(word) for word in text if not word in set(all_stopwords)]
        text = ' '.join(text)
        text = [text]        
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        cv = CountVectorizer(max_features = 400, max_df = 1)
        text = cv.fit_transform(text).toarray()
        
        model = joblib.load("svcmodel")
        pred = model.predict(text)
        print(pred)
        pred = pred[0]
        s = "The predicted catgeory is " + str(pred)
        return(render_template("index.html", result=s))
    else: 
        return(render_template("index.html", result="Predict 2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




