# 🤖 Customer Support Router — Agentic RAG System

> Intelligent Customer Support Agent powered by **Claude AI (Anthropic)** + **LangGraph** + **ChromaDB** + **Chainlit UI**

![Architecture](https://img.shields.io/badge/LLM-Claude_Sonnet_4-blueviolet)
![Framework](https://img.shields.io/badge/Framework-LangGraph-green)
![UI](https://img.shields.io/badge/UI-Chainlit-orange)
![Deploy](https://img.shields.io/badge/Deploy-Docker-blue)

---

## 📐 Architecture

```
User Query (Chainlit UI)
        │
        ▼
┌─────────────────┐
│ Categorize Query │ ──→ Technical / Billing / General
└────────┬────────┘
         │
         ▼
┌─────────────────────┐
│ Analyze Sentiment    │ ──→ Positive / Neutral / Negative
└────────┬────────────┘
         │
    ┌────┴─────────────────────────┐
    │        Smart Router          │
    ├──────┬──────┬──────┬─────────┤
    ▼      ▼      ▼      ▼
 🔧Tech  💳Bill  📋Gen  🚨Escalate
    │      │      │      │
    └──────┴──────┴──────┘
           │
    ┌──────┴──────┐
    │ ChromaDB    │ (RAG retrieval)
    │ Vector Store│
    └─────────────┘
           │
           ▼
    Final Response → User
```

## 🔧 Tech Stack

| Component | Technology | Notes |
|-----------|-----------|-------|
| **LLM** | Claude Sonnet 4 (Anthropic) | Replaces OpenAI GPT-4o-mini |
| **Orchestration** | LangGraph + LangChain | Agentic workflow with state management |
| **Vector Store** | ChromaDB | Persistent, cosine similarity |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | Free, local — no API key needed |
| **UI** | Chainlit | Rich chat interface with steps |
| **Deployment** | Docker + Docker Compose | Works on Windows, macOS, Linux |

---

## 🚀 Part 1: Local Deployment (Windows + Docker Desktop)

### Prerequisites

- **Docker Desktop** installed and running ([Download](https://www.docker.com/products/docker-desktop/))
- **Anthropic API Key** ([Get one here](https://console.anthropic.com/settings/keys))

### Step-by-Step

```bash
# 1. Clone/copy the project to your machine
cd customer-support-agent

# 2. Download the knowledge base (IMPORTANT!)
#    Download from: https://drive.google.com/file/d/1CWHutosAcJ6fiddQW5ogvg7NgLstZJ9j/view
#    Save as: data/router_agent_documents.json
#    (A sample file is already included — replace it with the full version for best results)

# 3. Create your .env file
copy .env.example .env
# Edit .env and paste your Anthropic API key:
#   ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# 4. Build and start with Docker Compose
docker compose up --build

# 5. Open your browser
#    → http://localhost:8000
```

### What Happens During First Start

1. Docker builds the image (~3-5 min first time)
2. Downloads the `all-MiniLM-L6-v2` embedding model (~90MB, cached)
3. Indexes the knowledge base into ChromaDB (persisted in Docker volume)
4. Starts Chainlit UI on port 8000

### Quick Commands

```bash
# Start in background
docker compose up -d

# View logs
docker compose logs -f

# Restart
docker compose restart

# Stop
docker compose down

# Rebuild after code changes
docker compose up -d --build

# Reset ChromaDB index (re-index from scratch)
docker volume rm customer-support-agent_chroma_data
docker compose up -d --build
```

### Troubleshooting (Local)

| Issue | Solution |
|-------|----------|
| Port 8000 in use | Change port in `docker-compose.yml`: `"8080:8000"` |
| API key error | Verify `.env` file has valid `ANTHROPIC_API_KEY` |
| Slow first start | Normal — embedding model download + indexing |
| Container keeps restarting | Check logs: `docker compose logs -f` |
| ChromaDB errors | Reset: `docker volume rm customer-support-agent_chroma_data` |

---

## ☁️ Part 2: AWS EC2 Deployment (Ubuntu VM)

### Step 1: Launch EC2 Instance

| Setting | Value |
|---------|-------|
| **AMI** | Ubuntu 22.04 LTS or 24.04 LTS |
| **Instance Type** | `t3.medium` (minimum: 2 vCPU, 4GB RAM) |
| **Storage** | 30 GB gp3 |
| **Security Group** | Allow TCP: **22** (SSH), **80** (HTTP), **443** (HTTPS) |
| **Key Pair** | Create or select an existing key pair |

### Step 2: Connect & Upload Project

```bash
# SSH into your EC2 instance
ssh -i your-key.pem ubuntu@YOUR_EC2_PUBLIC_IP

# Option A: Upload via SCP
scp -i your-key.pem -r ./customer-support-agent ubuntu@YOUR_EC2_PUBLIC_IP:~/

# Option B: Clone from Git (if you push to a repo)
git clone https://github.com/your-repo/customer-support-agent.git
```

### Step 3: Configure & Deploy

```bash
cd customer-support-agent

# Create .env with your API key
cp .env.example .env
nano .env
# Add: ANTHROPIC_API_KEY=sk-ant-api03-your-key-here

# Run the deployment script
chmod +x aws/deploy.sh
sudo ./aws/deploy.sh
```

The script automatically:
- Updates system packages
- Installs Docker & Docker Compose
- Installs & configures Nginx as reverse proxy
- Builds and starts the application

### Step 4: Access Your Application

```
http://YOUR_EC2_PUBLIC_IP
```

### Optional: Add HTTPS with Let's Encrypt

```bash
# Install Certbot
sudo apt install -y certbot python3-certbot-nginx

# Get SSL certificate (replace with your domain)
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
sudo certbot renew --dry-run
```

### EC2 Useful Commands

```bash
# View app logs
docker compose logs -f

# Restart app
docker compose restart

# Check Nginx status
sudo systemctl status nginx

# Check container status
docker ps

# Rebuild after code changes
docker compose up -d --build
```

---

## 📁 Project Structure

```
customer-support-agent/
├── app.py                          # Chainlit UI application
├── agent.py                        # LangGraph agent (Claude-powered)
├── knowledge_base_loader.py        # ChromaDB + HuggingFace embeddings
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container image definition
├── docker-compose.yml              # Multi-container orchestration
├── .env.example                    # Environment template
├── .dockerignore                   # Docker build exclusions
├── chainlit.md                     # Chainlit welcome page
├── .chainlit/
│   └── config.toml                 # Chainlit UI configuration
├── public/                         # Static assets (logos, CSS)
├── data/
│   └── router_agent_documents.json # Knowledge base (download/replace)
├── aws/
│   ├── deploy.sh                   # EC2 automated deployment
│   └── nginx.conf                  # Nginx reverse proxy config
└── README.md                       # This file
```

---

## 🔄 Key Changes from Original Notebook

| Original (Notebook) | This Application |
|---------------------|-----------------|
| OpenAI `gpt-4o-mini` | **Claude Sonnet 4** (`claude-sonnet-4-20250514`) |
| `OpenAIEmbeddings` | **HuggingFace** `all-MiniLM-L6-v2` (free, local) |
| Google Colab | **Docker + Chainlit** (local/cloud) |
| `OPENAI_API_KEY` | **`ANTHROPIC_API_KEY`** |
| `langchain-openai` | **`langchain-anthropic`** |
| IPython display | **Chainlit rich UI** with steps & badges |
| In-memory ChromaDB | **Persistent ChromaDB** (Docker volume) |

---

## 🧪 Sample Test Queries

| Query | Expected Category | Expected Sentiment |
|-------|------------------|--------------------|
| "Do you support pre-trained models?" | Technical | Neutral |
| "What payment methods do you accept?" | Billing | Neutral |
| "What is your refund policy?" | General | Neutral |
| "I'm fed up with this faulty hardware!" | Technical | Negative → Escalate |
| "What are your working hours?" | General | Neutral |

---

## 📄 License

This project is for educational and demonstration purposes.
