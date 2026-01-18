# ISKCON Kurnool – Website & My Krishna Assistant

Official website for **ISKCON Kurnool – Sri Sri Jagannath Mandir**
The site includes temple information pages and **My Krishna**, an AI assistant that answers questions based on *Bhagavad‑gītā As It Is*.

---

## Features

- Temple pages: Home, About, Deities, Visit, Programs, Calendar, Donate, Gallery, Contact.
- **My Krishna Q&A assistant**
  - Answers questions using Google Gemini.
  - Grounded on *Bhagavad‑gītā As It Is* via FAISS vector search.
  - Provides verse references with links to Bhaktivedanta Vedabase.
- **Contact form**
  - Frontend: Bootstrap form on `contact.html`.
  - Backend: FastAPI `/contact` endpoint.
  - Stores each submission in `contact_submissions.csv`.
  - Sends an email notification to the temple email.
- Multilingual text support (EN/TE/HI) using JSON translation files.
- Mobile‑responsive layout with Bootstrap 5.

---

## Tech Stack

- **Frontend**
  - HTML5, CSS3, Bootstrap 5
  - Vanilla JavaScript for API calls and i18n
- **Backend**
  - Python 3 + FastAPI
  - Uvicorn ASGI server
  - `google-genai` (Gemini API)
  - FAISS for semantic search over Gita verses
  - `python-dotenv` for configuration
- **Data**
  - `gita_meta.json` – verse text + metadata
  - FAISS index file for fast similarity search
  - `contact_submissions.csv` – stored contact messages (git‑ignored)

---

## Local Development

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your-username>/iskcon-kurnool-site.git
   cd iskcon-kurnool-site
Create and activate a virtualenv (optional but recommended)

bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
Install dependencies

bash
pip install -r requirements.txt
Create .env

text
GOOGLE_API_KEY=your_gemini_key

EMAIL_HOST=smtp.gmail.com
EMAIL_PORT=465
EMAIL_USER=yourgmail@gmail.com
EMAIL_PASS=your_app_password_without_spaces
EMAIL_TO=temple_destination_email@example.com
Run the FastAPI backend

bash
uvicorn main:app --reload --port 8000
Serve the frontend

Use a simple static server (for example, VS Code Live Server or Python):

bash
python -m http.server 5500
Then open:

http://127.0.0.1:5500/index.html for the main site.

http://127.0.0.1:5500/contact.html for the contact form.

Environment & Security
Secrets (Gemini API key, SMTP credentials) are loaded from .env.

.env and contact_submissions.csv are listed in .gitignore and must not be committed.

For production deployment (Render/Railway), move these values into platform environment variables.

API Endpoints (Backend)
POST /ask

Body: { "question": "string" }

Returns: AI answer and supporting Gita references.

POST /contact

Body: { "name", "phone", "email", "subject", "message" }

Behaviour: append row to CSV and send notification email.

Deployment (overview)
Container‑less deployment supported on platforms like Render or Railway:

Build: pip install -r requirements.txt

Start: uvicorn main:app --host 0.0.0.0 --port $PORT

Configure environment variables in the platform dashboard:

GOOGLE_API_KEY, EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS, EMAIL_TO.

Credits
Temple and spiritual guidance: ISKCON Kurnool – Sri Sri Jagannath Mandir.

Teachings: Bhagavad‑gītā As It Is by A.C. Bhaktivedanta Swami Prabhupāda (Bhaktivedanta Book Trust).

Developer: Naga Tharun N V.
