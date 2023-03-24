"""
Project: Ticket Sampling

"""

#Import all required Modules
import os
from flask import Flask, request,render_template,make_response
from werkzeug.utils import secure_filename
from keras.models import model_from_json
from tensorflow import keras
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pickle
from keras.optimizers import Adam
from keras.models import Model, Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Dropout

#Intialize Flask application
app = Flask(__name__)

#Store paths for upload directory, templates directory and retrain directory
uploads_dir = os.path.join(app.root_path, 'uploads')
templates_dir = os.path.join(app.root_path, 'templates')
retrain_dir = os.path.join(app.root_path, 'retrain')
os.makedirs(uploads_dir, exist_ok=True)

#Load Original Project Names from pickle file
enc_project_original = pickle.load(open(os.path.join(retrain_dir, "enc_project_original.pickle"), "rb"))
project_original = list(np.concatenate(enc_project_original.categories_).flat)

#Set loss list and test metrics for model evaluation
loss_list = ['categorical_crossentropy','categorical_crossentropy']
test_metrics = {'category': 'accuracy','priority': 'accuracy'}

#Currently model is trained on 50 Epochs, 0.0001 learning rate and 10 batch size
EPOCHS=50
@app.route('/')
#Home page for the Application
def home():
    # Load Encoder for Project List
    return render_template('prediction.html', original_project=project_original)

@app.route('/data_preparation')
def data_preparation(desc):
    tokens = []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    table = str.maketrans('', '', string.punctuation)

    for i in desc:
        token = word_tokenize(i)
        words = pd.Series(token).str.lower()
        words = [w.translate(table) for w in words]
        words = [w for w in words if w.lower() not in stop_words]
        words = pd.Series(words).replace('n', '')
        words = [w for w in words if w.isalpha()]
        words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(words)
        tokens.append(text)
    return np.array(tokens)

#Project preprocessing for training
@app.route('/project_preparation')
def project_preparation(project):
    if not project.empty:
        proj = []
        table = str.maketrans('', '', string.punctuation)
        for i in project:
            words = pd.Series(i).str.lower()
            words = [w.translate(table) for w in words]
            words = [w.replace(" ", "") for w in words]
            words = [w.strip() for w in words]
            txt = ' '.join(words)
            proj.append(txt)
        return pd.DataFrame(proj)

#Project preprocessing for prediction
@app.route('/pred_project_preparation')
def pred_project_preparation(project):
    project = project.lower()
    table = str.maketrans('', '', string.punctuation)
    project = project.translate(table)
    project = project.replace(" ", "")
    project = project.strip()
    return np.array(project)

#Route for Model Retrain template
@app.route("/temp_retrain")
def temp_retrain():
    return render_template('model_retrain.html')

#Route for Data Preview of Working Model
@app.route('/retrain_data_preview',methods=['GET', 'POST'])
def retrain_data_preview():
    f = open(os.path.join(retrain_dir,'file_name.txt'),"r")
    return render_template('retrain_data_preview.html', name=f.read())

#Route for html data table template (Newly loaded file)
@app.route('/data')
def data():
    return render_template('data.html')

#Route for html data table template
@app.route('/retrain_data')
def retrain_data():
    return render_template('retrain_data.html')

#Prediction
@app.route("/predict",methods=['POST'])
def predict():
    #Load Model and Dependencies
    json_file = open(os.path.join(retrain_dir,'cat_prior_model.json'), 'r')
    loaded_model_json = json_file.read()
    json_file.close()   
    multi_model = model_from_json(loaded_model_json)
    multi_model.load_weights(os.path.join(retrain_dir,'cat_prior_model.h5'))

    json_file1 = open(os.path.join(retrain_dir,'assign_model.json'), 'r')
    loaded_model_json1 = json_file1.read()
    json_file1.close()
    assign_model = model_from_json(loaded_model_json1)
    assign_model.load_weights(os.path.join(retrain_dir,'assign_model.h5'))

    vectorizer = pickle.load(open(os.path.join(retrain_dir, "vectorizer.pickle"), "rb"))
    enc_project = pickle.load(open(os.path.join(retrain_dir,"enc_project.pickle"), "rb"))
    enc_category = pickle.load(open(os.path.join(retrain_dir,"enc_category.pickle"), "rb"))
    enc_priority = pickle.load(open(os.path.join(retrain_dir,"enc_priority.pickle"), "rb"))
    enc_assign = pickle.load(open(os.path.join(retrain_dir,"enc_assign.pickle"),'rb'))

    #Request project and description from form 
    project = request.form['select2-single-box project']
    print(project)
    text = request.form['desc']
    print(text)
    #Pre-process & Encode the data
    prep_project = pred_project_preparation(project)
    vect_project = enc_project.transform(prep_project.reshape(-1,1)).toarray()
    prep_text = data_preparation([text])
    vect_desc = vectorizer.transform(prep_text).toarray()
    df = np.concatenate([vect_project,vect_desc],axis=1)
    #Makes a prediction of Category and Priority
    pred = multi_model.predict(df)

    #Decode prediction results
    category =enc_category.inverse_transform(pred[0])
    priority= enc_priority.inverse_transform(pred[1])

    #Prepare data for Assignment Prediction
    df1 = np.concatenate([vect_project,pred[0],pred[1],vect_desc],axis=1)
    #Assignment Prediction
    pred1 = assign_model.predict(df1)
    #Decode Result
    ass=enc_assign.inverse_transform(pred1)
    print('\n',category[0][0],'\n',priority[0][0],'\n',ass[0][0])
    return render_template('prediction.html',project=project, original_project=project_original, text=text, category=category[0][0],priority=priority[0][0],assign=ass[0][0], count = int(1))
    
#Callbacks at the time of training
class LossAndErrorPrintingCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        global EPOCHS, progress_count
        progress_count = int(((epoch+1)/EPOCHS)*100)
        response = make_response(render_template('model_retrain.html'))
        print('----------------------------------------' + str(progress_count) + '-------------------------------')
        return response

#Get no. of rows in choosen file
def show_rows(data):
    row=data.shape[0]
    return render_template('model_retrain.html', rows=row)

#Retrain Model
@app.route("/get_data",methods=['GET', 'POST'])
def get_data():
    if request.method == 'POST':
        global data, desc, project, category, priority,vectorizer,vect_desc,file_path,multi_model,assign_model, project_original
        global enc_project_original, enc_project, enc_category, enc_priority, enc_assign
        global trans_category, trans_priority, trans_assign ,trans_project, profile
        #path for upload directory
        path=os.path.join(uploads_dir,profile.filename)
        print(path)
        #check for csv and Excel file
        if path:
            if profile.filename.endswith('.csv'):
                data = pd.read_csv(path)
            elif profile.filename.endswith('.xlsx' or '.xls'):
                data = pd.read_excel(path)
            else:
                pass

            #initialize encoders
            enc_project_original = OneHotEncoder()
            enc_project = OneHotEncoder()
            enc_category = OneHotEncoder()
            enc_priority = OneHotEncoder()
            enc_assign = OneHotEncoder()
            #get data from selected file
            desc = data['desc']
            project = data['project']
            category = pd.DataFrame(data['category'])
            priority = pd.DataFrame(data['priority'])
            assign = pd.DataFrame(data['assign to'])
            #pre-processing 
            prep_data = data_preparation(desc)
            vectorizer = TfidfVectorizer()
            vect_desc = vectorizer.fit_transform(prep_data).toarray()
            prep_project = project_preparation(project)
            #One Hot Encoding
            trans_project_original = enc_project_original.fit_transform(pd.DataFrame(project))
            #Update Project list to preview in prediction page
            project_original = list(np.concatenate(enc_project_original.categories_).flat)
            trans_project = enc_project.fit_transform(prep_project).toarray()
            trans_category = enc_category.fit_transform(category).toarray()
            trans_priority = enc_priority.fit_transform(priority).toarray()
            trans_assign = enc_assign.fit_transform(assign).toarray()
            #prepare list of projects, categories, priority and assign
            d_x_cat_pri = {}
            d_y_cat_pri = {}
            d_x_ass = {}
            d_y_ass = {}
            pr = list(np.concatenate(enc_project.categories_).flat)
            cate = list(np.concatenate(enc_category.categories_).flat)
            prior = list(np.concatenate(enc_priority.categories_).flat)
            assign = list(np.concatenate(enc_assign.categories_).flat)

            #prepare dictonary assosiated with encoded values 
            x = [p for p in enumerate(pr)]
            y = [c for c in enumerate(cate)]
            z = [i for i in enumerate(prior)]
            ass = [i for i in enumerate(assign)]
            for i, j in x:
                d_x_cat_pri[j] = trans_project[:, i]
            for i, j in y:
                d_y_cat_pri[j] = trans_category[:, i]
            for i, j in z:
                d_y_cat_pri[j] = trans_priority[:, i]
            
            for i, j in x:
                d_x_ass[j] = trans_project[:, i]
            for i, j in y:
                d_x_ass[j] = trans_category[:, i]
            for i, j in z:
                d_x_ass[j] = trans_priority[:, i]
            for i, j in ass:
                d_y_ass[j] = trans_assign[:, i]
            #Dataframe for Category and Priority Prediction    
            d_x_cat_pri = pd.DataFrame(data=d_x_cat_pri)
            X_cat_pri = pd.concat([d_x_cat_pri, pd.DataFrame(vect_desc)], axis=1)
            Y_cat_pri = pd.DataFrame(data=d_y_cat_pri)
            #Split data for train and test
            X_train_duo, X_test_duo, Y_train_duo, Y_test_duo = train_test_split(X_cat_pri, Y_cat_pri, test_size=0.20, random_state=10)

            #Dataframe for Assignment Prediction 
            d_x_ass = pd.DataFrame(data=d_x_ass)
            X_ass = pd.concat([d_x_ass, pd.DataFrame(vect_desc)], axis=1)
            Y_ass = pd.DataFrame(data=d_y_ass)
            #Split data for train and test
            X_train_ass, X_test_ass, Y_train_ass, Y_test_ass = train_test_split(X_ass, Y_ass, test_size=0.20, random_state=10)
            
            # ---------------------Train--------------------------
            category_train = Y_train_duo[cate]
            category_nodes = category_train.shape[1]
            category_train = category_train.values
        
            priority_train = Y_train_duo[prior]
            priority_nodes = priority_train.shape[1]
            priority_train = priority_train.values
        
            # ---------------------Test--------------------------
            category_test = Y_test_duo[cate]
            category_nodes = category_test.shape[1]
            category_test = category_test.values
        
            priority_test = Y_test_duo[prior]
            priority_nodes = priority_test.shape[1]
            priority_test = priority_test.values
            
#-----------------------------------------Multi Model(Category & Priority)------------------------------------------------------------------------
            multi_model = Sequential()
            duo_model_input = Input(shape=(X_train_duo.shape[1],))
            x = multi_model(duo_model_input)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
                
            y1 = Dense(128, activation='relu')(x)
            y1 = Dropout(0.3)(y1)
            y1 = Dense(64, activation='relu')(y1)
            y1 = Dropout(0.3)(y1)
               
            y2 = Dense(128, activation='relu')(x)
            y2 = Dropout(0.3)(y2)
            y2 = Dense(64, activation='relu')(y2)
            y2 = Dropout(0.3)(y2)
               
            y1 = Dense(category_nodes, activation='softmax', name='category')(y1)
            y2 = Dense(priority_nodes, activation='softmax', name='priority')(y2)
                
            multi_model = Model(inputs=duo_model_input, outputs=[y1, y2])
            multi_model.compile(loss=loss_list, optimizer=Adam(lr=0.0001), metrics=test_metrics)
            
            multi_model.fit(x=X_train_duo, y=[category_train,priority_train], batch_size=10, epochs=EPOCHS,
                    validation_data=(X_test_duo,[category_test,priority_test]),callbacks=[LossAndErrorPrintingCallback()])

#------------------------------------------Assign Model---------------------------------------------------------------            
            
            ass_model = Sequential()
            ass_model_input = Input(shape=(X_train_ass.shape[1],))
            x = ass_model(ass_model_input)
            
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(64, activation='relu')(x)
            x = Dropout(0.3)(x)

            y1 = Dense(trans_assign.shape[1], activation='softmax', name='assign')(x)
            
            
            assign_model = Model(inputs=ass_model_input, outputs=[y1])
            assign_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])
                      
            assign_model.fit(x=X_train_ass, y=Y_train_ass, batch_size=10, epochs=EPOCHS,
                    validation_data=(X_test_ass,Y_test_ass),callbacks=[LossAndErrorPrintingCallback()])
            #Store Model and dependencies
            store()
            return render_template('model_retrain.html', complete=1, result='Your Model is Trained and Stored Successfully')
    
#Upload selected file
@app.route('/uploader', methods=['GET', 'POST'])
def uploader():
    global html_table_data
    if request.method == 'POST':
        global profile
        profile = request.files['file']
        path =os.path.join(uploads_dir,profile.filename)
        #Check for the CSV and Excel file
        if profile.filename.endswith('.csv'):
            #Save selected file
            profile.save(os.path.join(uploads_dir, secure_filename(profile.filename)))
            #Read File
            data=pd.read_csv(os.path.join(uploads_dir,secure_filename(profile.filename)))
        elif profile.filename.endswith('.xlsx' or '.xls'):
            #Save selected file
            profile.save(os.path.join(uploads_dir, secure_filename(profile.filename)))
            #Read File
            data = pd.read_excel(os.path.join(uploads_dir,secure_filename(profile.filename)))
        else:
            return render_template('model_retrain.html', msg=1)

        if data is not None:
            #Check for the file structure (Required columns are present or not)
            if ('project' not in data) or ('desc' not in data) or ('category' not in data) or ('priority' not in data) or ('assign to' not in data):
                return render_template('model_retrain.html', msg=1)
            else:
                #Set css for html table
                css="{{ url_for('static',filename='css/df_style.css') }}"
                pd.set_option('colheader_justify', 'center')
                # HTML Table for file data
                html_string = '''
                  <head>
                      <link rel="stylesheet" type="text/css" href="{css}">
                  </head>
                  <body>
                    {table}
                  </body>
                
                '''
                html_table_data = html_string.format(table=data.to_html(classes='mystyle'),css=css)
                # Output an HTML file
                with open(os.path.join(templates_dir,"data.html"), 'w') as f:
                    f.write(html_table_data)
                    f.close()

                return render_template('model_retrain.html', upload=1, f_name=profile.filename, row_data=data.shape[0])
          



def store():
    global html_table_data, profile
    # Store dependencies and models
    pickle.dump(vectorizer, open(os.path.join(retrain_dir,"vectorizer.pickle"), "wb"))
    pickle.dump(enc_project_original, open(os.path.join(retrain_dir, "enc_project_original.pickle"), "wb"))
    pickle.dump(enc_project, open(os.path.join(retrain_dir,"enc_project.pickle"), "wb"))
    pickle.dump(enc_category, open(os.path.join(retrain_dir,"enc_category.pickle"), "wb"))
    pickle.dump(enc_priority, open(os.path.join(retrain_dir,"enc_priority.pickle"), "wb"))
    pickle.dump(enc_assign, open(os.path.join(retrain_dir,"enc_assign.pickle"), "wb"))

    multi_model_json = multi_model.to_json()
    with open(os.path.join(retrain_dir,"cat_prior_model.json"), "w") as json_file:
        json_file.write(multi_model_json)
    multi_model.save_weights(os.path.join(retrain_dir,"cat_prior_model.h5"))
   
    assign_model_json = assign_model.to_json()
    with open(os.path.join(retrain_dir,"assign_model.json"), "w") as json_file:
        json_file.write(assign_model_json)
    assign_model.save_weights(os.path.join(retrain_dir,"assign_model.h5"))
    # Store file name of choosen file for further use
    with open(os.path.join(retrain_dir,"file_name.txt"), 'w') as f:
        f.write(profile.filename)
        f.close()
    # Output an HTML file
    with open(os.path.join(templates_dir,"retrain_data.html"), 'w') as f:
        f.write(html_table_data)
        f.close()

# Clear Browser Cache
def before_request():
    app.jinja_env.cache = {}


if __name__=='__main__':
    app.before_request(before_request)
    app.run(host='127.0.0.1' )





