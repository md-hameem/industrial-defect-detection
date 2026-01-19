# Web Application

Industrial Defect Detection web application.

## Structure

```
web/
├── backend/         # FastAPI backend
│   ├── main.py      # API endpoints
│   ├── inference.py # Model inference
│   └── requirements.txt
└── frontend/        # Next.js frontend
    └── src/app/     # React components
```

## Quick Start

### 1. Start Backend

```bash
cd web/backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

### 2. Start Frontend

```bash
cd web/frontend
npm install
npm run dev
```

### 3. Open Browser

- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## Default Login

- Username: `admin`
- Password: `secret`
