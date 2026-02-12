from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Files load karein
model = joblib.load('D:\Internship projects\student performance factor\Models\student_marks_predictor.pkl')
scaler = joblib.load('D:\Internship projects\student performance factor\Models\data_scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Sirf wahi 5 features lein jo Scaler mein fit hain
        num_features = [
            float(request.form['Hours_Studied']),
            float(request.form['Attendance']),
            float(request.form['Sleep_Hours']),
            float(request.form['Previous_Scores']),
            float(request.form['Tutoring_Sessions'])
        ]
        
        # 2. Baki 4 categorical features (Already Encoded inputs)
        cat_features = [
            float(request.form['Parental_Involvement']),
            float(request.form['Access_to_Resources']),
            float(request.form['Motivation_Level']),
            float(request.form['Teacher_Quality'])
        ]

        # 3. Numeric features ko scale karein (Jaise aapne notebook mein kiya)
        scaled_num = scaler.transform(np.array([num_features]))
        
        # 4. Final input combine karein (Scaled Num + Cat)
        # Sequence: Hours, Attend, Parental, Access, Sleep, Prev, Motivation, Tutor, Teacher
        final_input = np.array([[
            scaled_num[0][0], scaled_num[0][1], cat_features[0], 
            cat_features[1], scaled_num[0][2], scaled_num[0][3], 
            cat_features[2], scaled_num[0][4], cat_features[3]
        ]])

        prediction = model.predict(final_input)
        result = round(float(prediction[0]), 2)
        
        return render_template('index.html', result=result)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == "__main__":
    app.run(debug=True)