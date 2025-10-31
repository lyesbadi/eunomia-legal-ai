#!/bin/bash

################################################################################
# EUNOMIA - Oracle Linux 8 Setup Script
# Version: 1.0.0
# Description: Complete Docker infrastructure + security setup
# Author: EUNOMIA Technologies
# Date: 2025-01-30
#
# Requirements: Oracle Cloud Free Tier VM (4 OCPU / 24 GB RAM)
# OS: Oracle Linux 8
# Execution: sudo ./setup-oracle-linux.sh
################################################################################

set -euo pipefail  # Exit on error, undefined vars, pipe failures
IFS=$'\n\t'        # Better field splitting

# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

readonly SCRIPT_VERSION="1.0.0"
readonly SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
readonly LOG_FILE="/var/log/eunomia-setup.log"
readonly DOCKER_USER="eunomia"
readonly DOCKER_VERSION="24.0"
readonly DOCKER_COMPOSE_VERSION="2.24.0"
readonly TIMEZONE="Europe/Paris"
readonly SWAP_SIZE="8G"

# Colors for console output
readonly COLOR_RESET='\033[0m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_BLUE='\033[0;34m'

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

# Logging with timestamp
log() {
    local level="$1"
    shift
    local message="$*"
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case "$level" in
        INFO)
            echo -e "${COLOR_GREEN}[INFO]${COLOR_RESET} $message" | tee -a "$LOG_FILE"
            ;;
        WARN)
            echo -e "${COLOR_YELLOW}[WARN]${COLOR_RESET} $message" | tee -a "$LOG_FILE"
            ;;
        ERROR)
            echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $message" | tee -a "$LOG_FILE"
            ;;
        STEP)
            echo -e "\n${COLOR_BLUE}>> $message${COLOR_RESET}" | tee -a "$LOG_FILE"
            ;;
    esac
    
    echo "[$timestamp] [$level] $message" >> "$LOG_FILE"
}

# Check root privileges
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log ERROR "This script must be run with sudo or as root"
        exit 1
    fi
}

# Check OS distribution and version
check_os() {
    if [[ ! -f /etc/oracle-release ]]; then
        log ERROR "This script is designed for Oracle Linux 8 only"
        exit 1
    fi
    
    local version
    version=$(grep -oP '(?<=release )\d+' /etc/oracle-release)
    
    if [[ "$version" != "8" ]]; then
        log ERROR "Unsupported Oracle Linux version: $version (required: 8)"
        exit 1
    fi
    
    log INFO "Oracle Linux 8 detected"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ==============================================================================
# STEP 1: SYSTEM INITIALIZATION
# ==============================================================================

init_system() {
    log STEP "Step 1: System initialization"
    
    # Configure timezone
    log INFO "Setting timezone: $TIMEZONE"
    timedatectl set-timezone "$TIMEZONE" || {
        log ERROR "Failed to set timezone"
        exit 1
    }
    
    # System update
    log INFO "Updating system packages..."
    {
        dnf update -y
        dnf upgrade -y
    } &>> "$LOG_FILE" || {
        log ERROR "System update failed"
        exit 1
    }
    
    # Install essential tools
    log INFO "Installing essential system tools..."
    dnf install -y \
        git \
        curl \
        wget \
        htop \
        nano \
        vim \
        unzip \
        tar \
        net-tools \
        bind-utils \
        openssl \
        ca-certificates \
        gnupg \
        lsb-release \
        yum-utils \
        device-mapper-persistent-data \
        lvm2 &>> "$LOG_FILE" || {
        log ERROR "Failed to install system tools"
        exit 1
    }
    
    log INFO "System initialized successfully"
}

# ==============================================================================
# STEP 2: SWAP CONFIGURATION
# ==============================================================================

configure_swap() {
    log STEP "Step 2: Configuring SWAP ($SWAP_SIZE)"
    
    local swap_file="/swapfile"
    
    # Check if swap already exists
    if swapon --show | grep -q "$swap_file"; then
        log INFO "SWAP already configured, disabling..."
        swapoff "$swap_file"
        rm -f "$swap_file"
    fi
    
    # Create swap file
    log INFO "Creating SWAP file of $SWAP_SIZE..."
    fallocate -l "$SWAP_SIZE" "$swap_file" || {
        log ERROR "Failed to create swap file"
        exit 1
    }
    
    # Secure permissions
    chmod 600 "$swap_file"
    
    # Initialize and activate swap
    log INFO "Initializing and activating SWAP..."
    mkswap "$swap_file" &>> "$LOG_FILE"
    swapon "$swap_file"
    
    # Make persistent (add to /etc/fstab if not present)
    if ! grep -q "$swap_file" /etc/fstab; then
        echo "$swap_file none swap sw 0 0" >> /etc/fstab
        log INFO "SWAP added to /etc/fstab for persistence"
    fi
    
    # Verification
    local swap_active
    swap_active=$(swapon --show | grep "$swap_file" | awk '{print $3}')
    
    if [[ -n "$swap_active" ]]; then
        log INFO "SWAP activated: $swap_active"
    else
        log ERROR "SWAP verification failed"
        exit 1
    fi
}

# ==============================================================================
# STEP 3: KERNEL OPTIMIZATIONS FOR DOCKER
# ==============================================================================

optimize_kernel() {
    log STEP "Step 3: Kernel optimizations for Docker"
    
    local sysctl_conf="/etc/sysctl.d/99-eunomia-docker.conf"
    
    log INFO "Writing kernel parameters to $sysctl_conf..."
    
    cat > "$sysctl_conf" <<'EOF'
# ==============================================================================
# EUNOMIA - Kernel Optimizations for Docker + Elasticsearch
# ==============================================================================

# Elasticsearch / Qdrant requirement
vm.max_map_count=262144

# Network performance improvements
net.core.somaxconn=65535
net.core.netdev_max_backlog=5000
net.ipv4.tcp_max_syn_backlog=8192

# TCP optimizations
net.ipv4.tcp_tw_reuse=1
net.ipv4.tcp_fin_timeout=30
net.ipv4.tcp_keepalive_time=600
net.ipv4.tcp_keepalive_probes=5
net.ipv4.tcp_keepalive_intvl=15

# Open files limits
fs.file-max=2097152
fs.inotify.max_user_watches=524288
fs.inotify.max_user_instances=512

# Swappiness (prefer RAM)
vm.swappiness=10

# Dirty ratio (disk writes)
vm.dirty_ratio=15
vm.dirty_background_ratio=5

# Memory overcommit protection
vm.overcommit_memory=1

# Increase network buffer
net.core.rmem_max=16777216
net.core.wmem_max=16777216
net.ipv4.tcp_rmem=4096 87380 16777216
net.ipv4.tcp_wmem=4096 65536 16777216

# BBR congestion control (better performance)
net.core.default_qdisc=fq
net.ipv4.tcp_congestion_control=bbr
EOF

    # Apply immediately
    log INFO "Applying kernel parameters..."
    sysctl -p "$sysctl_conf" &>> "$LOG_FILE" || {
        log WARN "Some kernel parameters failed (non-critical)"
    }
    
    # Verify critical parameter
    local max_map_count
    max_map_count=$(sysctl -n vm.max_map_count)
    
    if [[ "$max_map_count" -ge 262144 ]]; then
        log INFO "vm.max_map_count = $max_map_count (OK for Qdrant)"
    else
        log ERROR "vm.max_map_count insufficient: $max_map_count"
        exit 1
    fi
    
    log INFO "Kernel optimizations applied"
}

# ==============================================================================
# STEP 4: DOCKER INSTALLATION
# ==============================================================================

install_docker() {
    log STEP "Step 4: Installing Docker $DOCKER_VERSION"
    
    # Check if Docker already installed
    if command_exists docker; then
        local current_version
        current_version=$(docker --version | grep -oP '\d+\.\d+\.\d+')
        log INFO "Docker already installed (version: $current_version)"
        
        # Ask for reinstallation confirmation
        read -p "Reinstall Docker? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log INFO "Docker installation skipped"
            return 0
        fi
        
        # Clean uninstall
        log INFO "Uninstalling existing Docker..."
        systemctl stop docker docker.socket containerd 2>/dev/null || true
        dnf remove -y docker-* containerd.io &>> "$LOG_FILE" || true
    fi
    
    # Add official Docker repository
    log INFO "Adding official Docker repository..."
    dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo &>> "$LOG_FILE" || {
        log ERROR "Failed to add Docker repository"
        exit 1
    }
    
    # Install Docker Engine
    log INFO "Installing Docker Engine + CLI + Containerd..."
    dnf install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io \
        docker-buildx-plugin \
        docker-compose-plugin &>> "$LOG_FILE" || {
        log ERROR "Docker installation failed"
        exit 1
    }
    
    # Start and enable Docker
    log INFO "Starting Docker service..."
    systemctl start docker
    systemctl enable docker &>> "$LOG_FILE"
    
    # Verify installation
    if ! docker --version &>/dev/null; then
        log ERROR "Docker installed but command not available"
        exit 1
    fi
    
    local installed_version
    installed_version=$(docker --version | grep -oP '\d+\.\d+\.\d+')
    log INFO "Docker installed: version $installed_version"
    
    # Functional test
    log INFO "Running Docker functional test (hello-world)..."
    if docker run --rm hello-world &>> "$LOG_FILE"; then
        log INFO "Docker is working correctly"
    else
        log ERROR "Docker test failed"
        exit 1
    fi
}

# ==============================================================================
# STEP 5: DOCKER COMPOSE INSTALLATION
# ==============================================================================

install_docker_compose() {
    log STEP "Step 5: Installing Docker Compose $DOCKER_COMPOSE_VERSION"
    
    # Docker Compose v2 is installed as plugin with Docker
    if docker compose version &>/dev/null; then
        local compose_version
        compose_version=$(docker compose version | grep -oP '\d+\.\d+\.\d+')
        log INFO "Docker Compose already installed: v$compose_version (plugin)"
        return 0
    fi
    
    log ERROR "Docker Compose plugin not detected after Docker installation"
    exit 1
}

# ==============================================================================
# STEP 6: NON-ROOT USER CREATION
# ==============================================================================

create_docker_user() {
    log STEP "Step 6: Creating non-root user '$DOCKER_USER'"
    
    # Check if user already exists
    if id "$DOCKER_USER" &>/dev/null; then
        log INFO "User '$DOCKER_USER' already exists"
        
        # Ensure user is in docker group
        if ! groups "$DOCKER_USER" | grep -q docker; then
            log INFO "Adding '$DOCKER_USER' to docker group..."
            usermod -aG docker "$DOCKER_USER"
        fi
    else
        # Create system user
        log INFO "Creating system user '$DOCKER_USER'..."
        useradd -r -m -s /bin/bash -G docker "$DOCKER_USER" || {
            log ERROR "Failed to create user"
            exit 1
        }
        
        # Add to wheel group (sudo)
        log INFO "Adding '$DOCKER_USER' to wheel group (sudo)..."
        usermod -aG wheel "$DOCKER_USER"
        
        # Generate secure random password
        local random_password
        random_password=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-20)
        echo "$DOCKER_USER:$random_password" | chpasswd
        
        log INFO "User created with random password"
        log WARN "Password for '$DOCKER_USER': $random_password"
        log WARN "SAVE THIS PASSWORD - It will not be shown again!"
        
        # Save credentials to secure file (root only)
        echo "User: $DOCKER_USER" > /root/eunomia-credentials.txt
        echo "Password: $random_password" >> /root/eunomia-credentials.txt
        echo "Created: $(date)" >> /root/eunomia-credentials.txt
        chmod 600 /root/eunomia-credentials.txt
        log INFO "Credentials saved in /root/eunomia-credentials.txt"
    fi
    
    # Create home directory if missing
    local user_home="/home/$DOCKER_USER"
    if [[ ! -d "$user_home" ]]; then
        mkdir -p "$user_home"
        chown -R "$DOCKER_USER:$DOCKER_USER" "$user_home"
    fi
    
    # Test Docker permissions
    log INFO "Testing Docker permissions for '$DOCKER_USER'..."
    if su - "$DOCKER_USER" -c "docker ps" &>> "$LOG_FILE"; then
        log INFO "'$DOCKER_USER' can use Docker without sudo"
    else
        log ERROR "Docker permissions test failed for '$DOCKER_USER'"
        exit 1
    fi
}

# ==============================================================================
# STEP 7: FIREWALL CONFIGURATION
# ==============================================================================

configure_firewall() {
    log STEP "Step 7: Configuring firewall (firewalld)"
    
    # Install firewalld if not present
    if ! command_exists firewall-cmd; then
        log INFO "Installing firewalld..."
        dnf install -y firewalld &>> "$LOG_FILE"
    fi
    
    # Start and enable firewalld
    log INFO "Enabling firewalld..."
    systemctl start firewalld
    systemctl enable firewalld &>> "$LOG_FILE"
    
    # Configure EUNOMIA ports
    log INFO "Opening required ports..."
    
    # Port 22 (SSH) - already open by default but ensure
    firewall-cmd --permanent --add-service=ssh &>> "$LOG_FILE"
    
    # Port 80 (HTTP)
    firewall-cmd --permanent --add-service=http &>> "$LOG_FILE"
    
    # Port 443 (HTTPS)
    firewall-cmd --permanent --add-service=https &>> "$LOG_FILE"
    
    # Reload firewall
    firewall-cmd --reload &>> "$LOG_FILE"
    
    # Verification
    log INFO "Verifying firewall configuration..."
    local open_services
    open_services=$(firewall-cmd --list-services)
    
    if [[ "$open_services" =~ ssh ]] && [[ "$open_services" =~ http ]] && [[ "$open_services" =~ https ]]; then
        log INFO "Firewall configured: $open_services"
    else
        log ERROR "Incomplete firewall configuration"
        exit 1
    fi
    
    # Display active rules
    log INFO "Active firewall rules:"
    firewall-cmd --list-all | tee -a "$LOG_FILE"
}

# ==============================================================================
# STEP 8: FAIL2BAN INSTALLATION & CONFIGURATION
# ==============================================================================

install_fail2ban() {
    log STEP "Step 8: Installing and configuring Fail2ban"
    
    # Install fail2ban
    log INFO "Installing Fail2ban..."
    dnf install -y fail2ban fail2ban-systemd &>> "$LOG_FILE" || {
        log ERROR "Fail2ban installation failed"
        exit 1
    }
    
    # Configure SSH jail
    log INFO "Configuring SSH jail..."
    cat > /etc/fail2ban/jail.d/sshd.local <<'EOF'
[sshd]
enabled = true
port = ssh
logpath = /var/log/secure
maxretry = 5
bantime = 3600
findtime = 600
banaction = firewallcmd-ipset
backend = systemd
EOF

    # Start and enable fail2ban
    log INFO "Starting Fail2ban..."
    systemctl start fail2ban
    systemctl enable fail2ban &>> "$LOG_FILE"
    
    # Verification
    sleep 2
    if systemctl is-active --quiet fail2ban; then
        log INFO "Fail2ban active and configured"
        
        # Display SSH jail status
        fail2ban-client status sshd 2>/dev/null | tee -a "$LOG_FILE" || true
    else
        log ERROR "Fail2ban failed to start"
        exit 1
    fi
}

# ==============================================================================
# STEP 9: CERTBOT INSTALLATION (LET'S ENCRYPT)
# ==============================================================================

install_certbot() {
    log STEP "Step 9: Installing Certbot (Let's Encrypt SSL)"
    
    # Install Certbot + NGINX plugin
    log INFO "Installing Certbot + NGINX plugin..."
    dnf install -y certbot python3-certbot-nginx &>> "$LOG_FILE" || {
        log ERROR "Certbot installation failed"
        exit 1
    }
    
    # Verification
    if command_exists certbot; then
        local certbot_version
        certbot_version=$(certbot --version 2>&1 | grep -oP '\d+\.\d+\.\d+')
        log INFO "Certbot installed: v$certbot_version"
    else
        log ERROR "Certbot installed but command not available"
        exit 1
    fi
    
    # Configure automatic renewal timer
    log INFO "Enabling automatic SSL renewal..."
    systemctl enable certbot-renew.timer &>> "$LOG_FILE" || {
        log WARN "certbot-renew timer not available (normal on Oracle Linux 8)"
    }
    
    # Instructions for obtaining certificate
    log INFO "-------------------------------------------------------"
    log INFO "To obtain SSL certificate after NGINX deployment:"
    log INFO "sudo certbot --nginx -d lyesbadii.xyz"
    log INFO "-------------------------------------------------------"
}

# ==============================================================================
# STEP 10: FINAL CHECKS
# ==============================================================================

final_checks() {
    log STEP "Step 10: Final verifications"
    
    local checks_passed=0
    local checks_total=10
    
    # Check 1: Docker
    if docker --version &>/dev/null; then
        log INFO "Docker operational"
        ((checks_passed++))
    else
        log ERROR "Docker not operational"
    fi
    
    # Check 2: Docker Compose
    if docker compose version &>/dev/null; then
        log INFO "Docker Compose operational"
        ((checks_passed++))
    else
        log ERROR "Docker Compose not operational"
    fi
    
    # Check 3: Docker User
    if su - "$DOCKER_USER" -c "docker ps" &>/dev/null; then
        log INFO "User '$DOCKER_USER' can use Docker"
        ((checks_passed++))
    else
        log ERROR "User '$DOCKER_USER' cannot use Docker"
    fi
    
    # Check 4: Firewall
    if firewall-cmd --state &>/dev/null; then
        log INFO "Firewall active"
        ((checks_passed++))
    else
        log ERROR "Firewall inactive"
    fi
    
    # Check 5: Fail2ban
    if systemctl is-active --quiet fail2ban; then
        log INFO "Fail2ban active"
        ((checks_passed++))
    else
        log ERROR "Fail2ban inactive"
    fi
    
    # Check 6: Certbot
    if command_exists certbot; then
        log INFO "Certbot installed"
        ((checks_passed++))
    else
        log ERROR "Certbot missing"
    fi
    
    # Check 7: SWAP
    if swapon --show | grep -q "/swapfile"; then
        log INFO "SWAP active"
        ((checks_passed++))
    else
        log ERROR "SWAP inactive"
    fi
    
    # Check 8: Kernel vm.max_map_count
    local max_map=$(sysctl -n vm.max_map_count)
    if [[ "$max_map" -ge 262144 ]]; then
        log INFO "Kernel optimized (vm.max_map_count=$max_map)"
        ((checks_passed++))
    else
        log ERROR "vm.max_map_count insufficient: $max_map"
    fi
    
    # Check 9: Timezone
    local current_tz
    current_tz=$(timedatectl | grep "Time zone" | awk '{print $3}')
    if [[ "$current_tz" == "$TIMEZONE" ]]; then
        log INFO "Timezone: $current_tz"
        ((checks_passed++))
    else
        log ERROR "Incorrect timezone: $current_tz"
    fi
    
    # Check 10: Open ports
    local open_ports
    open_ports=$(firewall-cmd --list-services)
    if [[ "$open_ports" =~ ssh ]] && [[ "$open_ports" =~ http ]] && [[ "$open_ports" =~ https ]]; then
        log INFO "Ports opened: $open_ports"
        ((checks_passed++))
    else
        log ERROR "Incorrect ports: $open_ports"
    fi
    
    # Final result
    echo ""
    log INFO "==========================================================="
    log INFO "  RESULT: $checks_passed/$checks_total checks passed"
    log INFO "==========================================================="
    echo ""
    
    if [[ $checks_passed -eq $checks_total ]]; then
        log INFO "All checks OK - System ready for EUNOMIA"
        return 0
    else
        log ERROR "Some checks failed - Review logs"
        return 1
    fi
}

# ==============================================================================
# CONFIGURATION SUMMARY DISPLAY
# ==============================================================================

display_summary() {
    log STEP "EUNOMIA configuration summary"
    
    cat <<EOF

==============================================================================
                    EUNOMIA - COMPLETE CONFIGURATION                      
==============================================================================

SYSTEM
   OS: Oracle Linux 8
   Timezone: $(timedatectl | grep "Time zone" | awk '{print $3}')
   Hostname: $(hostname)
   Public IP: $(curl -s ifconfig.me 2>/dev/null || echo "N/A")

DOCKER
   Version: $(docker --version | grep -oP '\d+\.\d+\.\d+')
   Compose: $(docker compose version | grep -oP '\d+\.\d+\.\d+')
   User: $DOCKER_USER (permissions OK)

SECURITY
   Firewall: $(firewall-cmd --state)
   Open ports: $(firewall-cmd --list-services)
   Fail2ban: $(systemctl is-active fail2ban)
   SSH jail: $(fail2ban-client status sshd 2>/dev/null | grep "Currently banned" | awk '{print $4}') banned IPs

RESOURCES
   Total RAM: $(free -h | grep Mem | awk '{print $2}')
   SWAP: $(free -h | grep Swap | awk '{print $2}')
   Root disk: $(df -h / | tail -1 | awk '{print $4}') available

KERNEL
   vm.max_map_count: $(sysctl -n vm.max_map_count)
   vm.swappiness: $(sysctl -n vm.swappiness)
   TCP congestion: $(sysctl -n net.ipv4.tcp_congestion_control)

CREDENTIALS
   User: $DOCKER_USER
   Password: See /root/eunomia-credentials.txt (root only)

NEXT STEPS

   1. Connect with non-root user:
      ssh $DOCKER_USER@$(curl -s ifconfig.me)

   2. Clone EUNOMIA repository:
      git clone https://github.com/your-repo/eunomia.git
      cd eunomia

   3. Configure .env:
      cp backend/.env.example backend/.env.production
      nano backend/.env.production  # Edit secrets

   4. Deploy Docker Compose:
      docker compose -f docker-compose.prod.yml up -d

   5. Obtain SSL certificate:
      sudo certbot --nginx -d yourdomain.com

LOGS
   Setup: $LOG_FILE
   Docker: journalctl -u docker -f
   Fail2ban: journalctl -u fail2ban -f

==============================================================================
  EUNOMIA - European Legal Order by AI                          
  https://eunomia.legal                                                   
==============================================================================

EOF
}

# ==============================================================================
# MAIN FUNCTION
# ==============================================================================

main() {
    # Banner
    cat <<'EOF'
==============================================================================
                                                                          
   ███████╗██╗   ██╗███╗   ██╗ ██████╗ ███╗   ███╗██╗ █████╗            
   ██╔════╝██║   ██║████╗  ██║██╔═══██╗████╗ ████║██║██╔══██╗           
   █████╗  ██║   ██║██╔██╗ ██║██║   ██║██╔████╔██║██║███████║           
   ██╔══╝  ██║   ██║██║╚██╗██║██║   ██║██║╚██╔╝██║██║██╔══██║           
   ███████╗╚██████╔╝██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║██║  ██║           
   ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝           
                                                                          
              European Legal Order by AI                        
                    Setup Script v1.0.0                                  
                                                                          
==============================================================================

EOF

    log INFO "Starting Oracle Linux 8 setup for EUNOMIA..."
    log INFO "Detailed logs: $LOG_FILE"
    echo ""
    
    # Pre-checks
    check_root
    check_os
    
    # User confirmation
    read -p "Continue with installation? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log INFO "Installation cancelled by user"
        exit 0
    fi
    
    # Sequential execution of steps
    local start_time
    start_time=$(date +%s)
    
    init_system
    configure_swap
    optimize_kernel
    install_docker
    install_docker_compose
    create_docker_user
    configure_firewall
    install_fail2ban
    install_certbot
    final_checks
    
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo ""
    log INFO "==========================================================="
    log INFO "  Installation completed in ${duration}s"
    log INFO "==========================================================="
    echo ""
    
    # Display summary
    display_summary
    
    log INFO "EUNOMIA setup completed successfully"
    log INFO "Next step: Deploy docker-compose.prod.yml"
}

# ==============================================================================
# ENTRY POINT
# ==============================================================================

# Create log file with root permissions
touch "$LOG_FILE" 2>/dev/null || {
    echo "Error: Cannot create $LOG_FILE (sudo required?)"
    exit 1
}

# Execute main
main "$@"

exit 0
