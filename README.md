# MAGICTales 🎭

**Speech Emotion Recognition & AI Story Generator**

A Flask-based web application that analyzes emotions from speech audio and generates personalized AI-powered stories based on detected emotions.

---

## 🌟 Features

- **Speech Emotion Recognition**: Detects 8 emotions (neutral, calm, happy, sad, angry, fear, disgust, surprise)
- **Audio Upload & Recording**: Support for multiple audio formats and live browser recording
- **AI Story Generation**: Creates personalized stories using Groq API based on detected emotions
- **Text-to-Speech**: Listen to generated stories with browser speech synthesis
- **User Management**: Registration, login, profile management, password reset
- **Admin Dashboard**: Analytics, user management, feedback system
- **Real-time Analytics**: Emotion detection statistics and visualizations

---

## 🛠️ Technology Stack

### Backend
- **Framework**: Flask 3.1.0
- **ML/AI**: PyTorch, Librosa, Groq API
- **Audio Processing**: Librosa, pydub, Web Audio API

### Frontend
- **Templates**: Jinja2, HTML5, CSS3
- **JavaScript**: Vanilla JS, Canvas API
- **Design**: Animated particle backgrounds, responsive UI

### Model
- **Architecture**: CRNN (Convolutional Recurrent Neural Network)
- **Features**: 40 MFCC coefficients
- **Sample Rate**: 22,050 Hz
- **Sequence Length**: 174 frames

---

## 📋 Prerequisites

- Python 3.7+
- pip (Python package manager)
- Modern web browser (Chrome, Firefox, Edge)
- Microphone (for live recording feature)

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/TreesaKuriakose/MAGICTales.git
cd MAGICTales/Speech-Emotion-Recognition
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
```bash
# Copy the example file
cp webapp/.env.example webapp/.env

# Edit .env and add your Groq API key
# Get your key from: https://console.groq.com/
```

### 5. Create Required Directories
```bash
mkdir -p webapp/uploads
```

### 6. Initialize Data Files (Optional)
Create empty JSON files if they don't exist:
```bash
echo "{}" > webapp/user_data.json
echo "{}" > webapp/emotion_analytics.json
echo "[]" > webapp/feedback_data.json
echo "{}" > webapp/password_reset_tokens.json
```

---

## ▶️ Running the Application

### Development Mode
```bash
cd webapp
python app.py
```

The application will be available at: `http://127.0.0.1:5000`

### Production Mode
```bash
# Use a production WSGI server like Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

---

## 👥 Demo Credentials

### Admin Account
- **Username**: `admin`
- **Password**: `admin123`
- **Access**: Full system administration

### Test User Accounts
- **Username**: `treesa` | **Password**: `treesa`
- **Username**: `meg` | **Password**: `meg@123M`

⚠️ **Security Note**: Change default passwords before deploying to production!

---

## 📁 Project Structure

```
Speech-Emotion-Recognition/
├── webapp/
│   ├── app.py                      # Main Flask application
│   ├── .env.example                # Environment variables template
│   ├── templates/                  # HTML templates
│   │   ├── index.html             # Home page
│   │   ├── dashboard.html         # User dashboard
│   │   ├── admin_dashboard.html   # Admin dashboard
│   │   ├── story.html             # Story generation page
│   │   ├── login.html             # Login page
│   │   ├── register.html          # Registration page
│   │   └── ...                    # Other templates
│   ├── uploads/                    # User uploaded audio files
│   ├── user_data.json             # User accounts (gitignored)
│   ├── emotion_analytics.json     # Analytics data (gitignored)
│   └── feedback_data.json         # User feedback (gitignored)
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

---

## 🎯 Usage Guide

### For Users

1. **Register/Login**: Create an account or login
2. **Upload Audio**: Go to Dashboard and upload an audio file (WAV, MP3, etc.)
3. **Or Record Live**: Use the browser recording feature
4. **View Results**: See the detected emotion
5. **Generate Story**: Click to generate an AI-powered story based on your emotion
6. **Listen**: Use text-to-speech to hear the story

### For Admins

1. **Login**: Use admin credentials
2. **View Analytics**: See emotion detection statistics
3. **Manage Users**: View, edit, or delete user accounts
4. **Review Feedback**: Read and respond to user feedback
5. **Monitor System**: Track usage patterns and performance

---

## 🎨 Supported Emotions

- 😐 **Neutral**: Calm, balanced emotional state
- 😌 **Calm**: Peaceful, relaxed state
- 😊 **Happy**: Joyful, positive emotions
- 😢 **Sad**: Sorrowful, melancholic feelings
- 😠 **Angry**: Frustrated, irritated state
- 😨 **Fear**: Anxious, worried emotions
- 🤢 **Disgust**: Repulsed, aversive feelings
- 😲 **Surprise**: Shocked, astonished reactions

---

## 🔧 Configuration

### Audio Settings
- **Sample Rate**: 22,050 Hz
- **MFCC Coefficients**: 40
- **Max Sequence Length**: 174 frames
- **Supported Formats**: WAV, MP3, OGG, FLAC, WebM, M4A, MP4

### API Configuration
Edit `webapp/.env`:
```env
GROQ_API_KEY=your_actual_groq_api_key
FLASK_SECRET_KEY=your_random_secret_key
```

---

## 🔒 Security Considerations

### For Development
- Default admin password is `admin123`
- User passwords stored in plain text (JSON file)
- No HTTPS encryption

### For Production (Recommended)
- ✅ Change all default passwords
- ✅ Implement password hashing (bcrypt)
- ✅ Use environment variables for secrets
- ✅ Enable HTTPS/SSL
- ✅ Use proper database (PostgreSQL/MySQL)
- ✅ Add rate limiting
- ✅ Implement CSRF protection
- ✅ Add session timeout
- ✅ Use JWT tokens

---

## 📊 Features Overview

### User Features
- Audio file upload (multiple formats)
- Live microphone recording
- Emotion detection with confidence scores
- AI story generation
- Text-to-speech narration
- Profile management
- Password reset
- Feedback submission

### Admin Features
- User management (CRUD operations)
- Emotion analytics dashboard
- Feedback management
- System statistics
- User activity monitoring

---

## 🐛 Troubleshooting

### Model Not Loading
- Ensure `ser_model.pth` exists in `load_model/` directory
- Check PyTorch installation
- Verify model architecture matches

### Audio Recording Not Working
- Grant microphone permissions in browser
- Use HTTPS (required for getUserMedia API)
- Check browser compatibility

### API Key Issues
- Verify `.env` file exists in `webapp/` directory
- Check Groq API key is valid
- Ensure `python-dotenv` is installed

---

## 📝 License

This project is for educational purposes. Please check individual dependencies for their licenses.

---

## 👨‍💻 Author

**Treesa Kuriakose**
- GitHub: [@TreesaKuriakose](https://github.com/TreesaKuriakose)

---

## 🙏 Acknowledgments

- RAVDESS dataset for emotion recognition training
- Groq API for AI story generation
- Flask framework and community
- PyTorch and Librosa libraries

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact the repository owner.

---

**⭐ If you find this project useful, please give it a star!**
