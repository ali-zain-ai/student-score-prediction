
import os
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import traceback
import secrets
import time


MODEL_PATH = r'D:\Internship projects\student performance factor\Models\student_marks_predictor.pkl'
SCALER_PATH = r'D:\Internship projects\student performance factor\Models\data_scaler.pkl'

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
app.config['JSON_SORT_KEYS'] = False


print("=" * 70)
print("  STUDENT PERFORMANCE PREDICTOR - Starting Application")
print("=" * 70)

try:
    model = joblib.load(MODEL_PATH)
    print(f"  ✅ Model loaded: {type(model).__name__}")
    print(f"  📁 Path: {MODEL_PATH}")
except Exception as e:
    print(f"  ❌ Model loading failed: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"  ✅ Scaler loaded: {type(scaler).__name__}")
    print(f"  📁 Path: {SCALER_PATH}")
except Exception as e:
    print(f"  ❌ Scaler loading failed: {e}")
    scaler = None

print("=" * 70)

# LabelEncoder order (alphabetical: High=0, Low=1, Medium=2)
CATEGORY_MAPPING = {
    'Parental_Involvement': {'High': 0, 'Low': 1, 'Medium': 2},
    'Access_to_Resources': {'High': 0, 'Low': 1, 'Medium': 2},
    'Motivation_Level': {'High': 0, 'Low': 1, 'Medium': 2},
    'Teacher_Quality': {'High': 0, 'Low': 1, 'Medium': 2}
}

# Reverse mapping for display
CATEGORY_DISPLAY = {
    'Parental_Involvement': {0: 'High', 1: 'Low', 2: 'Medium'},
    'Access_to_Resources': {0: 'High', 1: 'Low', 2: 'Medium'},
    'Motivation_Level': {0: 'High', 1: 'Low', 2: 'Medium'},
    'Teacher_Quality': {0: 'High', 1: 'Low', 2: 'Medium'}
}

# Feature columns in exact order used during training
FEATURE_COLUMNS = [
    'Hours_Studied',
    'Attendance',
    'Parental_Involvement',
    'Access_to_Resources',
    'Sleep_Hours',
    'Previous_Scores',
    'Motivation_Level',
    'Tutoring_Sessions',
    'Teacher_Quality'
]

# Numerical columns that need scaling
NUMERICAL_COLUMNS = [
    'Hours_Studied',
    'Attendance',
    'Sleep_Hours',
    'Previous_Scores',
    'Tutoring_Sessions'
]

# Validation rules based on dataset statistics
VALIDATION_RULES = {
    'Hours_Studied': {'min': 1, 'max': 44, 'default': 20, 'unit': 'hours/week'},
    'Attendance': {'min': 60, 'max': 100, 'default': 80, 'unit': '%'},
    'Previous_Scores': {'min': 50, 'max': 100, 'default': 75, 'unit': 'points'},
    'Sleep_Hours': {'min': 4, 'max': 10, 'default': 7, 'unit': 'hours/night'},
    'Tutoring_Sessions': {'min': 0, 'max': 8, 'default': 1, 'unit': 'sessions/week'}
}


def validate_input(data):
    """Validate and sanitize input data."""
    errors = []
    warnings = []
    
    # Required fields
    required_fields = [
        'hours_studied', 'attendance', 'parental_involvement',
        'access_to_resources', 'sleep_hours', 'previous_scores',
        'motivation_level', 'tutoring_sessions', 'teacher_quality'
    ]
    
    for field in required_fields:
        if field not in data or data[field] == '':
            errors.append(f"{field.replace('_', ' ').title()} is required")
    
    if errors:
        return False, errors, warnings
    
    # Validate numeric fields
    for field, rules in VALIDATION_RULES.items():
        field_key = field.lower()
        try:
            value = float(data[field_key])
            if value < rules['min']:
                warnings.append(f"{field} ({value}) is below minimum ({rules['min']} {rules['unit']})")
                data[field_key] = rules['min']
            elif value > rules['max']:
                warnings.append(f"{field} ({value}) exceeds maximum ({rules['max']} {rules['unit']})")
                data[field_key] = rules['max']
        except (ValueError, TypeError):
            errors.append(f"{field} must be a number")
            data[field_key] = rules['default']
    
    # Validate categorical fields
    categorical_fields = {
        'parental_involvement': 'Parental_Involvement',
        'access_to_resources': 'Access_to_Resources',
        'motivation_level': 'Motivation_Level',
        'teacher_quality': 'Teacher_Quality'
    }
    
    valid_categories = ['High', 'Low', 'Medium']
    for field_key, category in categorical_fields.items():
        value = data.get(field_key, 'Medium')
        if value not in valid_categories:
            errors.append(f"{field_key.replace('_', ' ').title()} must be High, Low, or Medium")
            data[field_key] = 'Medium'
    
    return len(errors) == 0, errors, warnings

def preprocess_features(data):
    """Transform raw input to model-ready features."""
    # Create DataFrame with exact column order and encoding
    df = pd.DataFrame([{
        'Hours_Studied': float(data['hours_studied']),
        'Attendance': float(data['attendance']),
        'Parental_Involvement': CATEGORY_MAPPING['Parental_Involvement'][data['parental_involvement']],
        'Access_to_Resources': CATEGORY_MAPPING['Access_to_Resources'][data['access_to_resources']],
        'Sleep_Hours': float(data['sleep_hours']),
        'Previous_Scores': float(data['previous_scores']),
        'Motivation_Level': CATEGORY_MAPPING['Motivation_Level'][data['motivation_level']],
        'Tutoring_Sessions': float(data['tutoring_sessions']),
        'Teacher_Quality': CATEGORY_MAPPING['Teacher_Quality'][data['teacher_quality']]
    }])
    
    # Ensure correct column order
    df = df[FEATURE_COLUMNS]
    
    # Scale numerical features using trained scaler
    df[NUMERICAL_COLUMNS] = scaler.transform(df[NUMERICAL_COLUMNS])
    
    return df

def generate_insights(features, predicted_score):
    """Generate personalized, actionable insights."""
    insights = []
    
    # Study hours insight
    hours = float(features['hours_studied'])
    if hours < 15:
        insights.append({
            'icon': '📚',
            'color': '#f59e0b',
            'title': 'Increase Study Hours',
            'message': f'You study {hours:.0f}h/week. Students with 20-25h score 8-12 points higher.',
            'action': 'Add 1 hour daily → +5 points potential'
        })
    elif hours > 30:
        insights.append({
            'icon': '⚡',
            'color': '#10b981',
            'title': 'Optimize Study Efficiency',
            'message': f'Great dedication ({hours:.0f}h)! Focus on quality over quantity.',
            'action': 'Try Pomodoro: 25min focus, 5min break'
        })
    else:
        insights.append({
            'icon': '✅',
            'color': '#3b82f6',
            'title': 'Optimal Study Hours',
            'message': f'Your study time ({hours:.0f}h) is in the sweet spot!',
            'action': 'Maintain consistency'
        })
    
    # Attendance insight
    attendance = float(features['attendance'])
    if attendance < 75:
        insights.append({
            'icon': '🏫',
            'color': '#ef4444',
            'title': 'Attendance Alert',
            'message': f'Your attendance ({attendance:.0f}%) is below average.',
            'action': 'Each 5% increase → +3-4 points'
        })
    elif attendance > 90:
        insights.append({
            'icon': '🎯',
            'color': '#10b981',
            'title': 'Excellent Attendance',
            'message': f'Outstanding! {attendance:.0f}% attendance is a top predictor of success.',
            'action': 'Keep it up!'
        })
    
    # Sleep insight
    sleep = float(features['sleep_hours'])
    if sleep < 6:
        insights.append({
            'icon': '😴',
            'color': '#ef4444',
            'title': 'Sleep Deprivation Risk',
            'message': f'You sleep {sleep:.1f}h. 7-8h improves memory by 40%.',
            'action': 'Sleep 1h earlier tonight'
        })
    elif sleep > 9:
        insights.append({
            'icon': '⏰',
            'color': '#f59e0b',
            'title': 'Sleep Schedule',
            'message': f'You sleep {sleep:.1f}h. Excess sleep may affect study time.',
            'action': 'Set consistent wake-up time'
        })
    
    # Tutoring insight
    tutoring = float(features['tutoring_sessions'])
    if tutoring == 0:
        insights.append({
            'icon': '🎓',
            'color': '#8b5cf6',
            'title': 'Try Tutoring',
            'message': 'You don\'t attend tutoring. 1-2 sessions/week adds 7-10 points!',
            'action': 'Join study group this week'
        })
    elif tutoring < 2:
        insights.append({
            'icon': '📝',
            'color': '#8b5cf6',
            'title': 'Increase Tutoring',
            'message': f'You attend {tutoring:.0f} session/week. 2-3 is optimal.',
            'action': 'Add one more session'
        })
    else:
        insights.append({
            'icon': '🌟',
            'color': '#10b981',
            'title': 'Great Initiative',
            'message': f'Regular tutoring ({tutoring:.0f}/week) strongly correlates with success.',
            'action': 'Continue this habit'
        })
    
    # Previous scores trend
    prev_score = float(features['previous_scores'])
    if predicted_score > prev_score + 5:
        insights.append({
            'icon': '📈',
            'color': '#10b981',
            'title': 'Improving!',
            'message': f'Predicted +{predicted_score - prev_score:.0f} points! Your habits are working.',
            'action': 'Document what changed'
        })
    elif predicted_score < prev_score - 5:
        insights.append({
            'icon': '🔄',
            'color': '#ef4444',
            'title': 'Slight Dip',
            'message': f'Predicted {prev_score - predicted_score:.0f} points lower.',
            'action': 'Review recent changes'
        })
    
    # Score-based motivation
    if predicted_score >= 80:
        insights.append({
            'icon': '🏆',
            'color': '#f59e0b',
            'title': 'Excellence Zone',
            'message': 'You\'re on track for top performance!',
            'action': 'Consider mentoring peers'
        })
    elif predicted_score >= 70:
        insights.append({
            'icon': '🎯',
            'color': '#3b82f6',
            'title': 'Above Average',
            'message': 'Solid performance! Small push to excellence.',
            'action': 'Focus on one weak area'
        })
    elif predicted_score >= 60:
        insights.append({
            'icon': '💪',
            'color': '#8b5cf6',
            'title': 'Good Foundation',
            'message': 'You have a solid base for improvement.',
            'action': 'Create weekly study plan'
        })
    else:
        insights.append({
            'icon': '🌱',
            'color': '#10b981',
            'title': 'Room to Grow',
            'message': 'Every point counts. Start with small changes.',
            'action': 'Set +5 point goal'
        })
    
    return insights[:4]  # Return top 4 insights

def get_performance_level(score):
    """Get performance level and color based on score."""
    if score >= 85:
        return {'label': 'Excellent', 'color': '#10b981', 'icon': '🏆'}
    elif score >= 75:
        return {'label': 'Very Good', 'color': '#3b82f6', 'icon': '🎯'}
    elif score >= 65:
        return {'label': 'Good', 'color': '#8b5cf6', 'icon': '✅'}
    elif score >= 55:
        return {'label': 'Satisfactory', 'color': '#f59e0b', 'icon': '📊'}
    else:
        return {'label': 'Needs Improvement', 'color': '#ef4444', 'icon': '⚠️'}


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html', result=None, form_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction from form data."""
    start_time = time.time()
    
    # Check if model is loaded
    if model is None or scaler is None:
        return render_template('index.html', 
                             error="Model not loaded. Please check model files.",
                             error_icon="🔴",
                             result=None,
                             form_data=None)
    
    try:
        # Get form data
        data = request.form.to_dict()
        
        # Validate input
        is_valid, errors, warnings = validate_input(data)
        if not is_valid:
            return render_template('index.html', 
                                 errors=errors,
                                 error_icon="⚠️",
                                 result=None,
                                 form_data=data)
        
        # Preprocess features
        features_df = preprocess_features(data)
        
        # Make prediction
        prediction = model.predict(features_df)[0]
        predicted_score = int(round(prediction))
        predicted_score = max(55, min(100, predicted_score))  # Clamp to valid range
        
        # Get performance level
        performance = get_performance_level(predicted_score)
        
        # Generate insights
        insights = generate_insights(data, predicted_score)
        
        # Calculate confidence score
        confidence = 85
        if warnings:
            confidence -= len(warnings) * 5
        confidence = max(70, min(95, confidence))
        
        # Prepare response
        result = {
            'score': predicted_score,
            'level': performance['label'],
            'level_color': performance['color'],
            'level_icon': performance['icon'],
            'confidence': confidence,
            'insights': insights,
            'warnings': warnings,
            'processing_time': f"{(time.time() - start_time)*1000:.0f}ms",
            'features': data
        }
        
        return render_template('index.html', result=result, form_data=data)
        
    except Exception as e:
        print(traceback.format_exc())
        return render_template('index.html', 
                             error="Something went wrong. Please try again.",
                             error_icon="🔴",
                             result=None,
                             form_data=data)

@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy' if model else 'degraded',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(e):
    return render_template('index.html', error="Page not found", error_icon="🔴", result=None, form_data=None)

@app.errorhandler(500)
def internal_error(e):
    return render_template('index.html', error="Server error", error_icon="🔴", result=None, form_data=None)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
