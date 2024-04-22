from flask import Flask, render_template, request, url_for, redirect
from os.path import join, dirname, realpath
import tensorflow as tf
from keras.models import load_model
import numpy as np
from flask_session import Session
import pdfkit
import uuid
import os
from datetime import date

app = Flask(__name__)
model = load_model('model.h5')
categories = ["adenocarinoma", "large cell carcinoma", "normal", "squamous cell carcinoma"]


#English
# recommendations = [
#     ["Schedule advanced imaging, such as PET-CT or MRI, to determine the extent and stage of the disease.",
#     "If not already done, consider a tissue biopsy to confirm the type and grade of the tumor and potentially assess for genetic mutations that may guide treatment.",
#     "Organize a tumor board or multidisciplinary team meeting, including a pulmonologist, oncologist, radiologist, and thoracic surgeon, to discuss the best course of action.",
#     "Based on further diagnostic tests, determine the TNM stage of the cancer to guide treatment decisions.",
#     "Develop a comprehensive treatment plan, which may include surgery, chemotherapy, radiation, targeted therapies, or immunotherapy."
#     ],
#     ["Undertake advanced imaging techniques, like PET-CT or MRI, to ascertain the stage and extent of the disease.",
#      "If not already completed, organize a tissue biopsy. This confirms the diagnosis, pinpoints the tumor type, and may help identify specific genetic or molecular characteristics of the tumor.",
#      "Convene a multidisciplinary team, including a pulmonologist, oncologist, radiologist, and thoracic surgeon, to deliberate on the best treatment approach.",
#      "Utilizing additional diagnostic tests, establish the TNM stage of the cancer, essential for treatment decisions.",
#      "Construct a treatment plan. Given the aggressive nature of large cell carcinoma, combined modalities such as surgery, chemotherapy, and radiation therapy might be necessary."],
#      ["Maintain a Smoke-Free Environment",
#       "Get Regular Check-ups"],
#     ["Schedule an appointment with a thoracic oncologist to discuss the diagnosis and establish a personalized treatment strategy.",
#      "Understand the variety of treatment methods available, which might include surgery, chemotherapy, radiation therapy, targeted therapies, or a combination of these.",
#      "Consider consulting another oncologist or a specialized cancer center to validate the diagnosis and suggested treatment.",
#      ]
# ]

 
#French
recommendations = [
    ["Planifiez des examens d'imagerie avancée, tels que la TEP-TDM ou l'IRM, pour déterminer l'étendue et le stade de la maladie.",
    "Si ce n'est pas déjà fait, envisagez une biopsie tissulaire pour confirmer le type et le grade de la tumeur et éventuellement évaluer les mutations génétiques qui peuvent guider le traitement.",
    "Organisez une réunion de concertation tumorale ou une réunion d'équipe multidisciplinaire, comprenant un pneumologue, un oncologue, un radiologue et un chirurgien thoracique, pour discuter du meilleur plan d'action.",
    "Sur la base de tests diagnostiques supplémentaires, déterminez le stade TNM du cancer pour orienter les décisions thérapeutiques.",
    "Élaborez un plan de traitement complet, qui peut inclure la chirurgie, la chimiothérapie, la radiothérapie, les thérapies ciblées ou l'immunothérapie."
    ],
    ["Entreprenez des techniques d'imagerie avancées, telles que la TEP-TDM ou l'IRM, pour déterminer le stade et l'étendue de la maladie.",
     "Si ce n'est pas déjà fait, organisez une biopsie tissulaire. Cela confirme le diagnostic, localise le type de tumeur et peut aider à identifier les caractéristiques génétiques ou moléculaires spécifiques de la tumeur.",
     "Convoquez une équipe multidisciplinaire, comprenant un pneumologue, un oncologue, un radiologue et un chirurgien thoracique, pour délibérer sur la meilleure approche thérapeutique.",
     "En utilisant des tests diagnostiques supplémentaires, établissez le stade TNM du cancer, essentiel pour les décisions thérapeutiques.",
     "Élaborez un plan de traitement. Compte tenu de la nature agressive du carcinome à grandes cellules, des modalités combinées telles que la chirurgie, la chimiothérapie et la radiothérapie peuvent être nécessaires."],
     ["Maintenez un environnement sans fumée",
      "Faites des bilans de santé réguliers"],
    ["Prenez rendez-vous avec un oncologue thoracique pour discuter du diagnostic et établir une stratégie de traitement personnalisée.",
     "Comprenez la variété des méthodes de traitement disponibles, qui peuvent inclure la chirurgie, la chimiothérapie, la radiothérapie, les thérapies ciblées ou une combinaison de celles-ci.",
     "Envisagez de consulter un autre oncologue ou un centre spécialisé dans le cancer pour valider le diagnostic et le traitement suggéré.",
     ]
]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def indexPost():
    userId = str(uuid.uuid4())
    UPLOADS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', userId)
    os.makedirs(UPLOADS_PATH)
    file = request.files['imagefile']
    file_type = file.filename.split('.')[-1]
    file.save(os.path.join(UPLOADS_PATH, 'scan.'+file_type))
    return redirect('/'+userId)

@app.route('/<uuid>', methods=['GET'])
def showImage(uuid):
    UPLOADS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'uploads', uuid)
    image_extensions = ['jpg', 'jpeg', 'png']
    image = None
    for ext in image_extensions:
        if os.path.exists(os.path.join(UPLOADS_PATH, 'scan.'+ext)):
            image = 'scan.'+ext
            break
    image_path = join(UPLOADS_PATH, image)
    image = url_for('static', filename='uploads/'+uuid+'/'+image)
    loaded_image =  tf.keras.utils.load_img(image_path, target_size=(224, 224))
    loaded_image = tf.keras.utils.img_to_array(loaded_image)
    loaded_image = loaded_image.reshape((1, loaded_image.shape[0], loaded_image.shape[1], loaded_image.shape[2]))
    predictions = model.predict(loaded_image)
    result = np.argmax(predictions, axis = 1)
    confidence = predictions[0][result]
    res = result[0]
    result = categories[result[0]]
    return render_template('response.html', image=image, result=result, confidence=confidence*100, recommendation = recommendations[res], date = date.today().strftime("%B %d, %Y"))

if __name__ == '__main__':
    app.run(port = 3000, debug=True)