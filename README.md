# 📞 AI Calling Agent

An AI-powered voice calling platform that enables real-time phone conversations using Large Language Models (LLMs), Speech-to-Text (STT), Text-to-Speech (TTS), and VoIP telephony integration.

The system can automatically answer calls, interact with users using natural language, handle customer inquiries, schedule appointments, and provide intelligent voice-based assistance.

---

## 🚀 Key Features

### 🤖 AI-Powered Conversations

* Natural language understanding
* Context-aware responses
* Real-time conversational AI
* Multi-turn dialogue support

### 📞 Voice Calling

* Outbound AI calls
* Inbound call handling
* Automated voice interactions
* VoIP integration

### 🎙️ Speech Processing

* Speech-to-Text (STT)
* Text-to-Speech (TTS)
* Real-time audio streaming

### 📅 Appointment & Booking Support

* Automated appointment scheduling
* Booking management
* Follow-up call automation

### 📊 Evaluation & Analytics

* Call performance evaluation
* Conversation quality analysis
* Response tracking

---

## 🏗️ System Architecture

```text
User Call
    │
    ▼
VoBIZ Telephony
    │
    ▼
Speech-to-Text Engine
    │
    ▼
OpenAI / LLM
    │
    ▼
Text-to-Speech Engine
    │
    ▼
Voice Response
```

---

## 📂 Project Structure

```text
AI-Calling-Agent/
│
├── voice_agent/
├── evaluation/
├── sample_doc/
├── static/
│
├── app.py
├── telephony_vobiz.py
├── setup_vobiz_trunk.py
├── make_vobiz_call.py
├── bookings.db
├── requirements.txt
└── README.md
```

---

## 🛠️ Technology Stack

### Backend

* Python
* Flask

### Artificial Intelligence

* OpenAI GPT Models
* Prompt Engineering

### Speech Technologies

* Speech-to-Text (STT)
* Text-to-Speech (TTS)

### Telephony

* VoBIZ SIP Trunk
* Voice Call Automation

### Database

* SQLite

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/broukrutha/Ai-Calling-Agent.git
cd Ai-Calling-Agent
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Configure Environment

Create a `.env` file and add:

```env
OPENAI_API_KEY=your_api_key
VOBIZ_API_KEY=your_api_key
```

### Run Application

```bash
python app.py
```

---

## 💡 Use Cases

### Customer Support

Automated customer service calls and query resolution.

### Appointment Scheduling

Book, reschedule, and confirm appointments automatically.

### Lead Qualification

Collect customer information and qualify leads through voice interactions.

### Healthcare Reminders

Automated medicine and appointment reminder calls.

### Educational Assistance

Voice-based student support and information services.

---

## 🔮 Future Enhancements

* Multilingual Support
* Emotion Detection
* Voice Cloning
* WhatsApp Calling
* CRM Integration
* Real-Time Analytics Dashboard
* Advanced Agent Memory
* Personalized AI Voices

---

## 🎯 Project Impact

This project demonstrates how conversational AI can automate voice-based interactions, reduce operational costs, and provide 24/7 intelligent assistance through phone calls.

---

## 👨‍💻 Developer

**R Broukrutha**

B.Tech Student

Gokaraju Rangaraju Institute of Engineering & Technology (GRIET)

---

⭐ If you find this project useful, consider giving it a Star.
