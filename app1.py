import numpy as np
import io
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import load_model
from flask import request
from flask import jsonify
from flask import Flask
from sklearn.externals import joblib
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

app = Flask(__name__)

def get_models():
    global model
    model = load_model('SaqmeFull.h5')
    print(" * Model loaded!")
    model1 = load_model('SaqmeFull_no_amenities.h5')
  
print(" * Loading Scaler...")
scaler = joblib.load('saqmeScaler')

print(" * Loading Keras model...")
get_models()


@app.route('/hello',methods=['POST'])
def hello():
    message = request.get_json(force=True)
    name = message['name']
    response = {
        'greeting':'Hello, ' + name + '!'
    }
    return jsonify(response)

@app.route('/predict',methods=['POST'])
def catchMessage():
    message = request.get_json(force=True)
    district = message['subDistrictId']
    area = message['totalSpace']
    rooms = message['rooms']
    floor = message['stage']
    n_floors = message['totalStage']
    heat = message['heating']
    gas = message['naturalGas']
    storage = message['storage']
    cellar = message['basement']
    balcony = message['balcony']
    garage = message['garage']
    #district = message['rooms']
    condition = message['state']
    status = message['status']
    project = message['project']

    #name = message['name']
    response = {
        'Price': 'Price is: ' + str(predict(area,rooms,floor,n_floors,heat,gas,storage,cellar,balcony,garage,district,condition,status,project))

    }
    return jsonify(response)


@app.route('/predict2', methods=['POST'])
def catchMessage_no_amenities():
    message = request.get_json(force=True)
    district = message['subDistrictId']
    area = message['totalSpace']
    rooms = message['rooms']
    floor = message['stage']
    n_floors = message['totalStage']
    condition = message['state']
    status = message['status']
    project = message['project']

    # name = message['name']
    response = {
        'Price': 'Price is: ' + str(predict_no_amenities(area, rooms, floor, n_floors, district, condition, status, project))

    }
    return jsonify(response)
    
def predict(area,rooms,floor,n_floors,heat,gas,storage,cellar,balcony,garage,district,condition,status,project):
    
    cols = ['Area', 'Rooms', 'Floor', 'Number_of_floors', 'Heat', 'Gas', 'Storage',
       'Cellar', 'Balcony', 'Garage', 'District_აეროპორტის გზატ.',
       'District_აეროპორტის დას.', 'District_ავლაბარი', 'District_ავჭალა',
       'District_აფრიკის დას', 'District_ბაგები', 'District_გლდანი',
       'District_გლდანულა', 'District_დამპალოს დას.', 'District_დიდი დიღომი',
       'District_დიდუბე', 'District_დიღმის მასივი', 'District_დიღომი 1-9',
       'District_ელია', 'District_ვაზისუბანი', 'District_ვაკე',
       'District_ვაჟა ფშაველას კვარტლები', 'District_ვარკეთილი',
       'District_ვაშლიჯვარი', 'District_ვერა', 'District_ვეძისი',
       'District_ზაჰესი', 'District_თბილისის ზღვა', 'District_თემქა',
       'District_თხინვალი', 'District_ისანი', 'District_კონიაკის დას.',
       'District_კუკია', 'District_კუს ტბა', 'District_ლილო',
       'District_ლისის ტბა', 'District_ლოტკინი', 'District_მესამე მასივი',
       'District_მთაწმინდა', 'District_მუხიანი', 'District_ნავთლუღი',
       'District_ნაძალადევი', 'District_ნუცუბიძის ფერდობი',
       'District_ორთაჭალა', 'District_ორხევი', 'District_საბურთალო',
       'District_სამგორი', 'District_სანზონა', 'District_სვანეთის უბანი',
       'District_სოლოლაკი', 'District_სოფ. გლდანი', 'District_სოფ. დიღომი',
       'District_ფონიჭალა', 'District_ჩუღურეთი', 'Condition_გარემონტებული',
       'Condition_თეთრი კარკასი', 'Condition_მიმდინარე რემონტი',
       'Condition_მწვანე კარკასი', 'Condition_სარემონტო',
       'Condition_შავი კარკასი', 'Condition_ძველი რემონტით',
       'Status_მშენებარე', 'Status_ძველი აშენებული', 'Project_დუპლექსი',
       'Project_ვეძისი', 'Project_თბილისური ეზო', 'Project_თუხარელის',
       'Project_იუგოსლავიის', 'Project_კიევი', 'Project_ლენინგრადის',
       'Project_ლვოვის', 'Project_მეტრომშენის', 'Project_მოსკოვის',
       'Project_სხვენი', 'Project_ტრიპექსი', 'Project_ქალაქური',
       'Project_ყავლაშვილის', 'Project_ჩეხური', 'Project_ხრუშჩოვის']
    
    example_data = pd.DataFrame(np.zeros((1,len(cols))),columns=cols)
    #example_data.drop('Price',axis=1,inplace=True)
    
    num = ['Area','Rooms','Floor','Number_of_floors']
    nums = [area,rooms,floor,n_floors]
    nums_scaled = scaler.transform([nums])
    #nums_scaled = numpy.array(nums_scaled)
    example_data[num] = nums_scaled
    
    amenities = ['Heat','Gas','Storage','Cellar','Balcony','Garage']
    amen = [heat,gas,storage,cellar,balcony,garage]
    example_data[amenities] = amen
    
    distr = 'District_'+district
    cond = 'Condition_'+condition
    stat = 'Status_'+status
    proj = 'Project_'+project
    
    if distr in example_data.columns:
        example_data[distr] = 1.0
        
    if cond in example_data.columns:
        example_data[cond] = 1.0
    
    if stat in example_data.columns:
        example_data[stat] = 1.0
    
    if proj in example_data.columns:
        example_data[proj] = 1.0
    
    
    #cat = pd.get_dummies(data,columns=cat, drop_first=True)
    for i in example_data.columns:
        print(i)
    #print(example_data)
    prediction = model.predict(example_data)
    #return example_data
    print(prediction)
    print(len(cols))
    return prediction[0][0]





def predict_no_amenities(area, rooms, floor, n_floors, district, condition, status,
            project):
    cols = ['Area', 'Rooms', 'Floor', 'Number_of_floors', 'District_აეროპორტის გზატ.',
            'District_აეროპორტის დას.', 'District_ავლაბარი', 'District_ავჭალა',
            'District_აფრიკის დას', 'District_ბაგები', 'District_გლდანი',
            'District_გლდანულა', 'District_დამპალოს დას.', 'District_დიდი დიღომი',
            'District_დიდუბე', 'District_დიღმის მასივი', 'District_დიღომი 1-9',
            'District_ელია', 'District_ვაზისუბანი', 'District_ვაკე',
            'District_ვაჟა ფშაველას კვარტლები', 'District_ვარკეთილი',
            'District_ვაშლიჯვარი', 'District_ვერა', 'District_ვეძისი',
            'District_ზაჰესი', 'District_თბილისის ზღვა', 'District_თემქა',
            'District_თხინვალი', 'District_ისანი', 'District_კონიაკის დას.',
            'District_კუკია', 'District_კუს ტბა', 'District_ლილო',
            'District_ლისის ტბა', 'District_ლოტკინი', 'District_მესამე მასივი',
            'District_მთაწმინდა', 'District_მუხიანი', 'District_ნავთლუღი',
            'District_ნაძალადევი', 'District_ნუცუბიძის ფერდობი',
            'District_ორთაჭალა', 'District_ორხევი', 'District_საბურთალო',
            'District_სამგორი', 'District_სანზონა', 'District_სვანეთის უბანი',
            'District_სოლოლაკი', 'District_სოფ. გლდანი', 'District_სოფ. დიღომი',
            'District_ფონიჭალა', 'District_ჩუღურეთი', 'Condition_გარემონტებული',
            'Condition_თეთრი კარკასი', 'Condition_მიმდინარე რემონტი',
            'Condition_მწვანე კარკასი', 'Condition_სარემონტო',
            'Condition_შავი კარკასი', 'Condition_ძველი რემონტით',
            'Status_მშენებარე', 'Status_ძველი აშენებული', 'Project_დუპლექსი',
            'Project_ვეძისი', 'Project_თბილისური ეზო', 'Project_თუხარელის',
            'Project_იუგოსლავიის', 'Project_კიევი', 'Project_ლენინგრადის',
            'Project_ლვოვის', 'Project_მეტრომშენის', 'Project_მოსკოვის',
            'Project_სხვენი', 'Project_ტრიპექსი', 'Project_ქალაქური',
            'Project_ყავლაშვილის', 'Project_ჩეხური', 'Project_ხრუშჩოვის']

    example_data = pd.DataFrame(np.zeros((1, len(cols))), columns=cols)
    # example_data.drop('Price',axis=1,inplace=True)

    num = ['Area', 'Rooms', 'Floor', 'Number_of_floors']
    nums = [area, rooms, floor, n_floors]
    nums_scaled = scaler.transform([nums])
    # nums_scaled = numpy.array(nums_scaled)
    example_data[num] = nums_scaled

    #amenities = ['Heat', 'Gas', 'Storage', 'Cellar', 'Balcony', 'Garage']
    #amen = [heat, gas, storage, cellar, balcony, garage]
    #example_data[amenities] = amen

    distr = 'District_' + district
    cond = 'Condition_' + condition
    stat = 'Status_' + status
    proj = 'Project_' + project

    if distr in example_data.columns:
        example_data[distr] = 1.0

    if cond in example_data.columns:
        example_data[cond] = 1.0

    if stat in example_data.columns:
        example_data[stat] = 1.0

    if proj in example_data.columns:
        example_data[proj] = 1.0

    # cat = pd.get_dummies(data,columns=cat, drop_first=True)
    #for i in example_data.columns:
     #   print(i)
    # print(example_data)
    prediction = model1.predict(example_data)
    # return example_data
    print(prediction)
    print(len(cols))
    return prediction[0][0]


if __name__ == "__main__":
	app.run()

