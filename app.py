import sys
import os
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch
import numpy as np
import librosa
from flask import Flask, request, render_template, redirect, url_for, session, send_from_directory
import json
from datetime import datetime
import re
import uuid
import smtplib
from email.mime.text import MIMEText
# Optional imports: fall back to stubs if modules or files are missing
try:
    from load_model.crnn import EmotionCRNN  # type: ignore
except Exception:
    EmotionCRNN = None  # type: ignore

try:
    from story_generator import generate_story_with_groq  # type: ignore
except Exception:
    generate_story_with_groq = None  # type: ignore

# Load environment variables
load_dotenv()

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'webm', 'm4a', 'mp4'}

SR = 22050
N_MFCC = 40
MAX_PAD_LEN = 174

EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']

# Get Groq API Key from environment variable (optional for local dev)
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'  # Needed for session

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load model if available, otherwise run in demo mode
model = None
if EmotionCRNN is not None:
    try:
        model = EmotionCRNN()
        model_path = os.path.join('..', 'load_model', 'ser_model.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
        else:
            model = None
    except Exception:
        model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def convert_audio_to_wav(audio_path):
    """Convert any audio file to WAV using pydub if available"""
    try:
        from pydub import AudioSegment
        wav_path = audio_path.rsplit('.', 1)[0] + '_converted.wav'
        
        # Detect format from file extension
        file_ext = audio_path.lower().split('.')[-1]
        if file_ext == 'webm':
            audio = AudioSegment.from_file(audio_path, format="webm")
        elif file_ext == 'mp4':
            audio = AudioSegment.from_file(audio_path, format="mp4")
        elif file_ext == 'ogg':
            audio = AudioSegment.from_file(audio_path, format="ogg")
        else:
            # Try auto-detection
            audio = AudioSegment.from_file(audio_path)
        
        audio.export(wav_path, format="wav")
        print(f"DEBUG: Converted {audio_path} to {wav_path}")
        return wav_path
    except ImportError:
        print("DEBUG: pydub not available for audio conversion")
        return None
    except Exception as e:
        print(f"DEBUG: Error converting audio: {str(e)}")
        return None

def extract_features(file_path, sr=SR, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    print(f"DEBUG: Loading audio file: {file_path}")
    original_path = file_path
    
    try:
        # If it's a format that librosa might have trouble with, try to convert it first
        file_ext = file_path.lower().split('.')[-1]
        if file_ext in ['webm', 'mp4', 'ogg']:
            converted_path = convert_audio_to_wav(file_path)
            if converted_path and os.path.exists(converted_path):
                file_path = converted_path
                print(f"DEBUG: Using converted file: {file_path}")
        
        audio, sample_rate = librosa.load(file_path, sr=sr)
        print(f"DEBUG: Audio loaded - length: {len(audio)}, sample_rate: {sample_rate}")
        
        if len(audio) == 0:
            raise ValueError("Audio file is empty or could not be loaded")
            
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        print(f"DEBUG: MFCC extracted - shape: {mfcc.shape}")
        
        if mfcc.shape[1] < max_pad_len:
            pad_width = max_pad_len - mfcc.shape[1]
            mfcc = np.pad(mfcc, pad_width=((0,0),(0,pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        
        print(f"DEBUG: Final MFCC shape: {mfcc.shape}")
        
        # Clean up converted file if we created one
        if file_path != original_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"DEBUG: Cleaned up converted file: {file_path}")
            except:
                pass
                
        return mfcc
    except Exception as e:
        print(f"DEBUG: Error in extract_features: {str(e)}")
        # Clean up converted file if we created one
        if file_path != original_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        raise

def update_emotion_analytics(emotion):
    """Update emotion analytics data for admin dashboard"""
    base_dir = os.path.dirname(__file__)
    emotion_path = os.path.join(base_dir, 'emotion_analytics.json')
    emotion_counts = _read_json_safe(emotion_path, {})
    
    # Increment count for this emotion
    current_count = emotion_counts.get(emotion, 0)
    emotion_counts[emotion] = current_count + 1
    
    # Save updated counts
    _write_json_safe(emotion_path, emotion_counts)

def update_story_analytics(emotion):
    """Update story generation analytics data for admin dashboard"""
    base_dir = os.path.dirname(__file__)
    story_path = os.path.join(base_dir, 'story_analytics.json')
    story_counts = _read_json_safe(story_path, {})
    
    # Increment count for this emotion's story
    current_count = story_counts.get(emotion, 0)
    story_counts[emotion] = current_count + 1
    
    # Save updated counts
    _write_json_safe(story_path, story_counts)

def analyze_saved_file(filepath):
    print(f"DEBUG: Analyzing file: {filepath}")
    try:
        mfcc = extract_features(filepath)
        print(f"DEBUG: MFCC shape: {mfcc.shape}")
        
        if model is not None:
            print("DEBUG: Using trained model for prediction")
            x = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                output = model(x)
                pred = torch.argmax(output, dim=1).item()
                emotion = EMOTION_LABELS[pred]
        else:
            print("DEBUG: Using fallback prediction method")
            mean_val = float(np.mean(mfcc))
            idx = int(abs(mean_val * 100)) % len(EMOTION_LABELS)
            emotion = EMOTION_LABELS[idx]
        
        print(f"DEBUG: Predicted emotion: {emotion}")
        
        # Update emotion analytics for admin dashboard
        update_emotion_analytics(emotion)
        
        return emotion
    except Exception as e:
        print(f"DEBUG: Error in analyze_saved_file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if not username or not password:
            error = 'Please enter both username and password.'
        else:
            # Admin login shortcut: if admin credentials are used on user login page,
            # go directly to admin dashboard and do not treat as a normal user.
            if username.strip().lower() == 'admin' and password == 'admin123':
                session.pop('user', None)
                session['admin'] = 'admin'
                return redirect(url_for('admin_dashboard'))
            # Normal user login: only allow registered users
            base_dir = os.path.dirname(__file__)
            users_path = os.path.join(base_dir, 'user_data.json')
            users = _read_json_safe(users_path, {})
            user_info = users.get(username)
            if not user_info:
                error = 'User not found. Please register first.'
            else:
                stored_pw = user_info.get('password')
                if stored_pw is None:
                    error = 'This account has no password set. Please register again.'
                elif stored_pw != password:
                    error = 'Incorrect password.'
                else:
                    # Mark logged-in status for admin visibility
                    user_info['status'] = 'Online'
                    user_info['is_logged_in'] = True
                    users[username] = user_info
                    _write_json_safe(users_path, users)
                    session.pop('admin', None)
                    session['user'] = username
                    return redirect(url_for('dashboard'))
    return render_template('login.html', error=error)

@app.route('/profile', methods=['GET', 'POST'])
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    error = None
    message = None
    profile_pic = session.get('profile_pic', None)
    
    # Get user bio from user_data.json
    base_dir = os.path.dirname(__file__)
    users_path = os.path.join(base_dir, 'user_data.json')
    users = _read_json_safe(users_path, {})
    username = session['user']
    user_bio = users.get(username, {}).get('bio', '')
    
    if request.method == 'POST':
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file.filename:
                pic_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{session['user']}_profile.png")
                file.save(pic_path)
                session['profile_pic'] = f"{session['user']}_profile.png"
                message = 'Profile picture updated.'
        elif 'new_password' in request.form:
            new_pw = request.form.get('new_password', '')
            ok, err = validate_password_strength(new_pw)
            if not ok:
                error = err
            else:
                if username in users:
                    users[username]['password'] = new_pw
                    _write_json_safe(users_path, users)
                    message = 'Password updated successfully.'
        elif 'edit_profile' in request.form:
            bio = request.form.get('edit_profile', '')
            if username in users:
                users[username]['bio'] = bio
                _write_json_safe(users_path, users)
                user_bio = bio
                message = 'Profile updated successfully.'
    return render_template('profile.html', user=session['user'], profile_pic=profile_pic, bio=user_bio, error=error, message=message)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('dashboard.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('dashboard.html', error='No selected file')
        if file and allowed_file(file.filename):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            try:
                emotion = analyze_saved_file(filepath)
                session['last_emotion'] = emotion
                return render_template('dashboard.html', emotion=emotion)
            except Exception as e:
                return render_template('dashboard.html', error=f'Error processing file: {e}')
        else:
            return render_template('dashboard.html', error='Invalid file type')
    # Check for last emotion from recording session and display it
    last_emotion = session.pop('last_emotion', None)
    return render_template('dashboard.html', emotion=last_emotion)

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    if 'user' not in session:
        return { 'error': 'Unauthorized' }, 401
    if 'file' not in request.files:
        return { 'error': 'No file part' }, 400
    file = request.files['file']
    filename = file.filename or 'recording.wav'
    
    if not allowed_file(filename):
        return { 'error': f'Invalid file type: {filename}. Allowed: {", ".join(ALLOWED_EXTENSIONS)}' }, 400
    try:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        emotion = analyze_saved_file(filepath)
        session['last_emotion'] = emotion
        return { 'emotion': emotion }
    except Exception as e:
        return { 'error': str(e) }, 500

@app.route('/story', methods=['GET', 'POST'])
def story():
    if 'user' not in session:
        return redirect(url_for('login'))
    emotion = session.get('last_emotion', None)
    story = None
    if emotion:
        try:
            if generate_story_with_groq is not None and GROQ_API_KEY:
                story = generate_story_with_groq(emotion, api_key=GROQ_API_KEY)
            else:
                # Fallback offline story generator - 15+ sentences
                story = (
                    f"Once upon a time, in a realm where emotions painted the very fabric of reality, "
                    f"there lived a {emotion} traveler whose heart carried the weight of countless unspoken words. "
                    f"This wanderer, whose name was whispered only by the wind, had journeyed through valleys of doubt "
                    f"and mountains of uncertainty, seeking a place where their {emotion} spirit could find true belonging. "
                    f"One fateful evening, as the stars began their ancient dance across the velvet sky, "
                    f"the traveler stumbled upon a hidden gateway adorned with symbols that seemed to pulse with living light. "
                    f"Beyond this mystical portal lay the enchanted land of Magictales, a realm where every feeling, "
                    f"no matter how complex or contradictory, was celebrated as a precious gift from the universe itself. "
                    f"As the {emotion} wanderer stepped through the threshold, they felt their heart begin to resonate "
                    f"with the very essence of this magical world, as if the land itself recognized the depth of their emotional journey. "
                    f"The trees whispered ancient wisdom in languages that spoke directly to the soul, "
                    f"while rivers sang melodies that seemed to understand the unspoken yearnings of the heart. "
                    f"Guided by their {emotion} nature, the traveler discovered that this realm held no judgment, "
                    f"only acceptance and the promise of transformation through emotional authenticity. "
                    f"Each step forward revealed new wonders: flowers that bloomed in response to genuine feelings, "
                    f"crystals that amplified the power of honest expression, and creatures that offered companionship "
                    f"without demanding the traveler to be anything other than their true, {emotion} self. "
                    f"In this sacred space, the wanderer learned that their {emotion} heart was not a burden to carry, "
                    f"but a compass that would guide them toward their ultimate destiny. "
                    f"And so, with courage born from self-acceptance and kindness that flowed from emotional honesty, "
                    f"the once-lost traveler found not just a path, but a purpose that would illuminate their journey "
                    f"toward a happily-ever-after written in the stars themselves."
                )
            # Track story generation for analytics
            update_story_analytics(emotion)
        except Exception as e:
            story = f"Could not generate story: {str(e)}"
    return render_template('story.html', emotion=emotion, story=story)

@app.route('/feedback', methods=['GET', 'POST'])
def feedback():
    if 'user' not in session:
        return redirect(url_for('login'))
    base_dir = os.path.dirname(__file__)
    feedback_path = os.path.join(base_dir, 'feedback_data.json')
    all_feedback = _read_json_safe(feedback_path, [])

    success = False
    if request.method == 'POST':
        rating = request.form.get('rating')
        text = request.form.get('feedback')
        try:
            rating_int = int(rating)
        except Exception:
            rating_int = 0
        if rating_int < 1 or rating_int > 5 or not text:
            # Re-render with error
            user_items = [f for f in all_feedback if f.get('user') == session['user']]
            return render_template('feedback.html', error='Please provide rating 1-5 and feedback text.', success=False, feedback=user_items)

        new_id = (max([int(item.get('id', 0)) for item in all_feedback]) + 1) if all_feedback else 1
        all_feedback.append({
            'id': new_id,
            'user': session['user'],
            'rating': rating_int,
            'feedback': text,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'admin_reply': None
        })
        _write_json_safe(feedback_path, all_feedback)
        success = True

    # Show ALL feedback from all users (sorted by newest first)
    all_feedback_sorted = sorted(all_feedback, key=lambda x: x.get('timestamp', ''), reverse=True)
    return render_template('feedback.html', success=success, feedback=all_feedback_sorted, current_user=session['user'])

@app.route('/logout')
def logout():
    # Update status in user_data when a normal user logs out
    base_dir = os.path.dirname(__file__)
    users_path = os.path.join(base_dir, 'user_data.json')
    if 'user' in session:
        username = session['user']
        users = _read_json_safe(users_path, {})
        if username in users:
            users[username]['status'] = 'Offline'
            users[username]['is_logged_in'] = False
            _write_json_safe(users_path, users)
        session.pop('user', None)
    return redirect(url_for('index'))

# ---------- Admin Helpers ----------
def _read_json_safe(path, default_value):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default_value

def _write_json_safe(path, data):
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False

def validate_password_strength(password: str):
    if len(password) < 8:
        return False, 'Password must be at least 8 characters.'
    if not re.search(r'[A-Z]', password):
        return False, 'Password must include an uppercase letter.'
    if not re.search(r'[a-z]', password):
        return False, 'Password must include a lowercase letter.'
    if not re.search(r'\d', password):
        return False, 'Password must include a number.'
    if not re.search(r'[^A-Za-z0-9]', password):
        return False, 'Password must include a special character.'
    return True, None

def send_password_reset_email(to_email: str, reset_link: str) -> bool:
    host = os.getenv('SMTP_HOST')
    port = int(os.getenv('SMTP_PORT', '0') or 0)
    user = os.getenv('SMTP_USER')
    password = os.getenv('SMTP_PASS')
    from_email = os.getenv('SMTP_FROM') or user
    subject = 'MagicTales Password Reset'
    body = f'Click the link to reset your password: {reset_link}'
    if host and port and user and password and from_email:
        try:
            msg = MIMEText(body)
            msg['Subject'] = subject
            msg['From'] = from_email
            msg['To'] = to_email
            with smtplib.SMTP(host, port) as server:
                server.starttls()
                server.login(user, password)
                server.sendmail(from_email, [to_email], msg.as_string())
            return True
        except Exception:
            pass
    try:
        base_dir = os.path.dirname(__file__)
        out_path = os.path.join(base_dir, 'password_reset_links.txt')
        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(f'{datetime.now().isoformat()} | {to_email} | {reset_link}\n')
        return True
    except Exception:
        return False

def _require_admin():
    if 'admin' not in session:
        return redirect(url_for('login'))
    return None

# ---------- Admin Routes ----------
@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username and password and username.strip().lower() == 'admin' and password == 'admin123':
            session.pop('user', None)
            session['admin'] = 'admin'
            return redirect(url_for('admin_dashboard'))
        error = 'Invalid admin credentials.'
    return render_template('admin_login.html', error=error)

@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect(url_for('index'))

@app.route('/admin/dashboard')
def admin_dashboard():
    guard = _require_admin()
    if guard is not None:
        return guard

    base_dir = os.path.dirname(__file__)
    emotion_path = os.path.join(base_dir, 'emotion_analytics.json')
    users_path = os.path.join(base_dir, 'user_data.json')

    emotion_counts = _read_json_safe(emotion_path, {})
    users = _read_json_safe(users_path, {})

    # Compute totals and stats
    total_users = len(users) if isinstance(users, dict) else 0
    total_emotions = sum(int(v) for v in emotion_counts.values()) if isinstance(emotion_counts, dict) else 0

    most_detected = ("—", 0)
    least_detected = ("—", 0)
    emotion_stats = None

    if isinstance(emotion_counts, dict) and emotion_counts:
        # Build stats with percentages
        emotion_stats = {}
        for emotion, count in emotion_counts.items():
            c = int(count)
            percentage = (c / total_emotions * 100.0) if total_emotions > 0 else 0.0
            emotion_stats[emotion] = {
                'count': c,
                'percentage': percentage
            }
        # Determine most/least detected
        most_detected = max(emotion_counts.items(), key=lambda item: int(item[1]))
        least_detected = min(emotion_counts.items(), key=lambda item: int(item[1]))

    return render_template(
        'admin_dashboard.html',
        total_users=total_users,
        total_emotions=total_emotions,
        most_detected=most_detected,
        least_detected=least_detected,
        emotion_stats=emotion_stats
    )

@app.route('/admin/profile', methods=['GET', 'POST'])
def admin_profile():
    guard = _require_admin()
    if guard is not None:
        return guard
    
    error = None
    message = None
    profile_pic = session.get('admin_profile_pic', None)
    
    # Get admin bio from admin_data.json
    base_dir = os.path.dirname(__file__)
    admin_data_path = os.path.join(base_dir, 'admin_data.json')
    admin_data = _read_json_safe(admin_data_path, {})
    admin_bio = admin_data.get('bio', '')
    
    if request.method == 'POST':
        # Handle profile picture upload
        if 'profile_pic' in request.files:
            file = request.files['profile_pic']
            if file.filename:
                pic_path = os.path.join(app.config['UPLOAD_FOLDER'], 'admin_profile.png')
                file.save(pic_path)
                session['admin_profile_pic'] = 'admin_profile.png'
                message = 'Profile picture updated successfully.'
        
        # Handle password change
        elif 'new_password' in request.form:
            new_pw = request.form.get('new_password', '')
            ok, err = validate_password_strength(new_pw)
            if not ok:
                error = err
            else:
                # Update admin password in a secure storage (for now, update in memory/session)
                # In production, store this in a database with proper hashing
                admin_data['password'] = new_pw
                _write_json_safe(admin_data_path, admin_data)
                message = 'Password updated successfully.'
        
        # Handle profile edit
        elif 'edit_profile' in request.form:
            bio = request.form.get('edit_profile', '')
            admin_data['bio'] = bio
            _write_json_safe(admin_data_path, admin_data)
            admin_bio = bio
            message = 'Profile updated successfully.'
    
    return render_template('profile.html', user='admin', profile_pic=profile_pic, bio=admin_bio, error=error, message=message)

@app.route('/admin/users')
def admin_users():
    guard = _require_admin()
    if guard is not None:
        return guard
    base_dir = os.path.dirname(__file__)
    users_path = os.path.join(base_dir, 'user_data.json')
    users = _read_json_safe(users_path, {})
    return render_template('admin_users.html', users=users)

@app.route('/admin/users/<username>')
def admin_edit_user(username):
    guard = _require_admin()
    if guard is not None:
        return guard
    base_dir = os.path.dirname(__file__)
    users_path = os.path.join(base_dir, 'user_data.json')
    users = _read_json_safe(users_path, {})
    user_info = users.get(username, { 'emotions': [], 'profile_pic': None })
    return render_template('admin_edit_user.html', username=username, user_info=user_info)

@app.route('/admin/users/<username>', methods=['POST'])
def admin_update_user(username):
    guard = _require_admin()
    if guard is not None:
        return guard
    base_dir = os.path.dirname(__file__)
    users_path = os.path.join(base_dir, 'user_data.json')
    users = _read_json_safe(users_path, {})
    user_info = users.get(username, { 'emotions': [], 'profile_pic': None })
    # Update minimal fields present in template
    user_info['bio'] = request.form.get('bio', user_info.get('bio', ''))
    user_info['status'] = request.form.get('status', user_info.get('status', 'Active'))
    users[username] = user_info
    _write_json_safe(users_path, users)
    return redirect(url_for('admin_edit_user', username=username))

@app.route('/admin/feedback')
def admin_feedback():
    guard = _require_admin()
    if guard is not None:
        return guard
    base_dir = os.path.dirname(__file__)
    feedback_path = os.path.join(base_dir, 'feedback_data.json')
    feedback = _read_json_safe(feedback_path, [])
    return render_template('admin_feedback.html', feedback=feedback)

@app.route('/admin/feedback/<int:feedback_id>', methods=['POST'])
def admin_reply_feedback(feedback_id):
    guard = _require_admin()
    if guard is not None:
        return guard
    base_dir = os.path.dirname(__file__)
    feedback_path = os.path.join(base_dir, 'feedback_data.json')
    feedback = _read_json_safe(feedback_path, [])
    reply_text = request.form.get('reply')
    for item in feedback:
        if int(item.get('id', -1)) == feedback_id:
            item['admin_reply'] = {
                'admin': 'admin',
                'text': reply_text,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            break
    _write_json_safe(feedback_path, feedback)
    return redirect(url_for('admin_feedback'))

@app.route('/admin/visualization')
def admin_visualization():
    guard = _require_admin()
    if guard is not None:
        return guard
    
    base_dir = os.path.dirname(__file__)
    emotion_path = os.path.join(base_dir, 'emotion_analytics.json')
    story_path = os.path.join(base_dir, 'story_analytics.json')
    
    emotion_counts = _read_json_safe(emotion_path, {})
    story_counts = _read_json_safe(story_path, {})
    
    return render_template('admin_visualization.html', 
                         emotion_data=emotion_counts, 
                         story_data=story_counts)

@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        if not username or not email or not password:
            error = 'Please fill out all fields.'
        else:
            base_dir = os.path.dirname(__file__)
            users_path = os.path.join(base_dir, 'user_data.json')
            users = _read_json_safe(users_path, {})
            if username in users:
                error = 'Username already exists. Please choose another.'
            else:
                ok, err = validate_password_strength(password)
                if not ok:
                    return render_template('register.html', error=err)
                users[username] = {
                    'email': email,
                    'password': password,
                    'emotions': [],
                    'profile_pic': None,
                    'status': 'Offline',
                    'is_logged_in': False
                }
                _write_json_safe(users_path, users)
                return redirect(url_for('login'))
    return render_template('register.html', error=error)

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    error = None
    message = None
    if request.method == 'POST':
        email = request.form.get('email')
        if not email:
            error = 'Please enter your email.'
        else:
            base_dir = os.path.dirname(__file__)
            users_path = os.path.join(base_dir, 'user_data.json')
            users = _read_json_safe(users_path, {})
            username = None
            for u, info in users.items():
                if info.get('email') == email:
                    username = u
                    break
            if not username:
                error = 'No account found with that email.'
            else:
                token = uuid.uuid4().hex
                tokens_path = os.path.join(base_dir, 'password_reset_tokens.json')
                tokens = _read_json_safe(tokens_path, {})
                tokens[token] = {
                    'username': username,
                    'email': email,
                    'created_at': datetime.now().isoformat()
                }
                _write_json_safe(tokens_path, tokens)
                reset_link = request.host_url.rstrip('/') + url_for('reset_password', token=token)
                if send_password_reset_email(email, reset_link):
                    message = 'Password reset link sent to your email.'
                else:
                    error = 'Could not send email. Please try again later.'
    return render_template('forgot_password.html', error=error, message=message)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    base_dir = os.path.dirname(__file__)
    tokens_path = os.path.join(base_dir, 'password_reset_tokens.json')
    tokens = _read_json_safe(tokens_path, {})
    token_info = tokens.get(token)
    if not token_info:
        return 'Invalid or expired token.', 400
    error = None
    message = None
    if request.method == 'POST':
        new_pw = request.form.get('password')
        ok, err = validate_password_strength(new_pw or '')
        if not ok:
            error = err
        else:
            users_path = os.path.join(base_dir, 'user_data.json')
            users = _read_json_safe(users_path, {})
            username = token_info['username']
            if username in users:
                users[username]['password'] = new_pw
                _write_json_safe(users_path, users)
                tokens.pop(token, None)
                _write_json_safe(tokens_path, tokens)
                return redirect(url_for('login'))
    return render_template('reset_password.html', error=error, message=message)

if __name__ == '__main__':
    app.run(debug=True) 