# EUNOMIA - European Legal Order by AI

> *Sovereign GDPR-compliant Legal AI Platform*

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-blue.svg)](https://react.dev/)
[![Docker](https://img.shields.io/badge/Docker-24.0%2B-blue.svg)](https://www.docker.com/)

## Overview

EUNOMIA is a European-sovereign legal AI platform combining:
- **80% Hugging Face models** (Legal-BERT, CamemBERT-NER, Unfair-ToS detector)
- **20% Ollama Mistral 7B** (custom recommendations generation)
- **100% GDPR compliance** (on-premise, EU-hosted, zero data sharing)

### Triple Meaning
- **Historical**: Greek concept from Solon (594 BC) = "Good legal order"
- **Geographic**: EU = European Union (digital sovereignty)
- **Semantic**: EU (good) + NOMOS (law) = "The Good Law"

## Architecture
```
7-Layer Architecture:
├─ Layer 1: NGINX (reverse proxy, SSL, rate limiting)
├─ Layer 2: FastAPI x2 (load balanced REST API)
├─ Layer 3: AI Services (Hugging Face models)
├─ Layer 4: LLM (Ollama Mistral 7B Q8)
├─ Layer 5: Celery Workers (async processing)
├─ Layer 6: Data Stores (PostgreSQL, Qdrant, Redis)
└─ Layer 7: Observability (Prometheus, Grafana, Loki)
```

**Total RAM**: 22.85 GB / 24 GB (Oracle Cloud Free Tier)

## Quick Start

### Prerequisites
- Oracle Cloud Free Tier VM (4 OCPU, 24 GB RAM)
- Oracle Linux 8
- Domain name (for SSL certificate)

### 1. Setup Infrastructure
```bash
# SSH into your Oracle VM
ssh opc@<YOUR_VM_IP>

# Clone repository
git clone https://github.com/YOUR-USERNAME/eunomia-legal-ai.git
cd eunomia-legal-ai

# Run setup script
sudo ./scripts/setup-oracle-linux.sh
```

This script installs:
- Docker + Docker Compose
- Firewall configuration (ports 80, 443, 22)
- Fail2ban (SSH protection)
- Certbot (Let's Encrypt SSL)
- Kernel optimizations for Docker
- 8GB SWAP

### 2. Configure Environment
```bash
# Copy environment template
cp backend/.env.example backend/.env.production

# Edit secrets (CHANGE ALL PASSWORDS!)
nano backend/.env.production
```

### 3. Deploy Stack
```bash
# Deploy all services
docker compose -f docker-compose.prod.yml up -d

# Wait for models download (~30-60 min first time)
docker logs -f legal-api-1

# Obtain SSL certificate
sudo certbot --nginx -d yourdomain.com
```

### 4. Verify Deployment
```bash
# Check all services running
docker ps

# Access API docs
open https://yourdomain.com/docs

# Access monitoring
open https://yourdomain.com:3000  # Grafana
```

## Technology Stack

### Backend
- **Python 3.11+** / FastAPI / SQLAlchemy 2.0 (async)
- **AI**: Hugging Face Transformers + Ollama
- **Queue**: Celery + Redis
- **Databases**: PostgreSQL + Qdrant (vectors) + Redis

### Frontend
- **React 18** + TypeScript / Vite
- **UI**: Tailwind CSS + shadcn/ui
- **State**: Zustand / TanStack Query
- **Router**: React Router v6

### Infrastructure
- **OS**: Oracle Linux 8
- **Containers**: Docker + Docker Compose
- **Proxy**: NGINX + Let's Encrypt SSL
- **Monitoring**: Prometheus + Grafana + Loki

## AI Models Used

### Hugging Face Models (80% of tasks)
- **Legal-BERT** (nlpaueb/legal-bert-base-uncased) - Document classification
- **CamemBERT-NER** (Jean-Baptiste/camembert-ner) - French NER
- **Unfair-ToS** (coastalcph/unfair-tos) - Abusive clauses detection
- **Sentence Transformers** (all-MiniLM-L6-v2) - Embeddings
- **BART** (facebook/bart-large-cnn) - Summarization
- **RoBERTa** (deepset/roberta-base-squad2) - Q&A

### Ollama Models (20% of tasks)
- **Mistral 7B Instruct Q8** - Custom recommendations, clause generation

## Development

### Local Development Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements-dev.txt

# Run tests
pytest

# Run locally
uvicorn app.main:app --reload

# Frontend
cd frontend
npm install
npm run dev
```

### Project Structure
```
eunomia-legal-ai/
├── backend/          # FastAPI application
├── frontend/         # React application
├── scripts/          # Deployment scripts
├── nginx/            # NGINX configuration
├── postgres/         # DB initialization
├── monitoring/       # Prometheus, Grafana configs
└── docs/             # Documentation
```

## Roadmap

- [x] **Phase 1** (Weeks 1-4): MVP Core
  - [x] Infrastructure setup
  - [x] Backend API + AI pipeline
  - [x] Frontend basic UI
- [ ] **Phase 2** (Weeks 5-8): Advanced Features
  - [ ] GDPR compliance (anonymization, audit logs)
  - [ ] Ollama integration
  - [ ] Semantic search
- [ ] **Phase 3** (Weeks 9-10): Monitoring & Ops
  - [ ] Observability stack
  - [ ] CI/CD pipeline
- [ ] **Phase 4** (Weeks 11-12): Go-to-Market
  - [ ] Landing page
  - [ ] Stripe integration
  - [ ] Pilot program

## License

**Proprietary Software - All Rights Reserved**

Copyright (c) 2025 [TON NOM]. This project is private and confidential.

Unauthorized copying, distribution, or use is strictly prohibited.


## Contact

- Email: [chougarlyes12@gmail.com]
- GitHub: [@lyesbadii](https://github.com/lyesbadii)

---

**EUNOMIA** - *European Legal Order by AI*  
Private Project - All Rights Reserved
