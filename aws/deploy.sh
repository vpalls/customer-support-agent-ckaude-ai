#!/bin/bash
# ============================================================
# AWS EC2 Ubuntu Deployment Script
# Customer Support Agent — Claude + LangGraph + Chainlit
# ============================================================
#
# USAGE:
#   1. Launch an Ubuntu 22.04/24.04 EC2 instance (t3.medium+ recommended)
#   2. SSH into the instance
#   3. Upload or clone this project
#   4. Run: chmod +x aws/deploy.sh && sudo ./aws/deploy.sh
#
# PREREQUISITES:
#   - EC2 Security Group: Allow inbound TCP 80, 443, 22
#   - EC2 Instance: t3.medium or larger (2 vCPU, 4GB RAM minimum)
#   - Ubuntu 22.04 or 24.04 LTS
# ============================================================

set -euo pipefail

echo "============================================"
echo "🚀 Customer Support Agent — EC2 Deployment"
echo "============================================"

# ── 1. System Updates ──
echo ""
echo "📦 Step 1/6: Updating system packages..."
apt-get update -y
apt-get upgrade -y

# ── 2. Install Docker ──
echo ""
echo "🐳 Step 2/6: Installing Docker..."
if ! command -v docker &> /dev/null; then
    # Install Docker using official script
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh

    # Add ubuntu user to docker group
    usermod -aG docker ubuntu

    # Enable Docker on boot
    systemctl enable docker
    systemctl start docker
    echo "   ✅ Docker installed"
else
    echo "   ✅ Docker already installed"
fi

# ── 3. Install Docker Compose ──
echo ""
echo "🔧 Step 3/6: Installing Docker Compose..."
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    COMPOSE_VERSION=$(curl -s https://api.github.com/repos/docker/compose/releases/latest | grep -Po '"tag_name": "\K.*?(?=")')
    curl -L "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    chmod +x /usr/local/bin/docker-compose
    echo "   ✅ Docker Compose installed (${COMPOSE_VERSION})"
else
    echo "   ✅ Docker Compose already installed"
fi

# ── 4. Install Nginx ──
echo ""
echo "🌐 Step 4/6: Installing Nginx..."
apt-get install -y nginx
systemctl enable nginx

# Configure Nginx
cp aws/nginx.conf /etc/nginx/sites-available/support-agent
ln -sf /etc/nginx/sites-available/support-agent /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default

# Test and reload Nginx
nginx -t
systemctl reload nginx
echo "   ✅ Nginx configured"

# ── 5. Setup Environment ──
echo ""
echo "🔑 Step 5/6: Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo ""
    echo "   ⚠️  IMPORTANT: Edit the .env file with your Anthropic API key!"
    echo "   Run: nano .env"
    echo ""
else
    echo "   ✅ .env file already exists"
fi

# ── 6. Build and Start ──
echo ""
echo "🏗️  Step 6/6: Building and starting containers..."
docker compose build --no-cache
docker compose up -d

echo ""
echo "============================================"
echo "✅ Deployment Complete!"
echo "============================================"
echo ""
echo "📍 Access your app at:"
echo "   http://$(curl -s http://169.254.169.254/latest/meta-data/public-ipv4 2>/dev/null || echo 'YOUR_EC2_PUBLIC_IP'):80"
echo ""
echo "📋 Useful commands:"
echo "   docker compose logs -f          # View logs"
echo "   docker compose restart          # Restart app"
echo "   docker compose down             # Stop app"
echo "   docker compose up -d --build    # Rebuild & restart"
echo ""
echo "🔑 Don't forget to set your API key in .env!"
echo "============================================"
