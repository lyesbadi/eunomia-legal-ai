#!/bin/bash

# ==============================================================================
# EUNOMIA - SCRIPT MAÎTRE DE TEST COMPLET
# Prépare l'environnement et lance tous les tests automatiquement
# ==============================================================================

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# ==============================================================================
# BANNER
# ==============================================================================

clear
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}"
echo "  ███████╗██╗   ██╗███╗   ██╗ ██████╗ ███╗   ███╗██╗ █████╗ "
echo "  ██╔════╝██║   ██║████╗  ██║██╔═══██╗████╗ ████║██║██╔══██╗"
echo "  █████╗  ██║   ██║██╔██╗ ██║██║   ██║██╔████╔██║██║███████║"
echo "  ██╔══╝  ██║   ██║██║╚██╗██║██║   ██║██║╚██╔╝██║██║██╔══██║"
echo "  ███████╗╚██████╔╝██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║██║  ██║"
echo "  ╚══════╝ ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝╚═╝  ╚═╝"
echo -e "${NC}"
echo "                  🇪🇺 Legal AI Platform - Test Suite 🇪🇺"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${BLUE}📍 Répertoire du projet : ${NC}$PROJECT_ROOT"
echo -e "${BLUE}📍 Répertoire des scripts : ${NC}$SCRIPT_DIR"
echo ""

# ==============================================================================
# FONCTION : AFFICHER UNE ÉTAPE
# ==============================================================================

print_step() {
    echo ""
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}$1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# ==============================================================================
# ÉTAPE 1 : VÉRIFICATION DE L'ENVIRONNEMENT
# ==============================================================================

print_step "📋 ÉTAPE 1/6 : Vérification de l'environnement"

echo ""
echo "🔍 Vérification des dépendances..."

# Vérifier curl
if ! command -v curl &> /dev/null; then
    echo -e "${RED}❌ curl n'est pas installé${NC}"
    echo "   Installation : sudo yum install -y curl"
    exit 1
fi
echo -e "${GREEN}✅ curl installé${NC}"

# Vérifier jq
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}⚠️  jq n'est pas installé (optionnel mais recommandé)${NC}"
    echo "   Installation : sudo yum install -y jq"
    echo ""
    read -p "Voulez-vous installer jq maintenant ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo yum install -y jq
        echo -e "${GREEN}✅ jq installé${NC}"
    fi
else
    echo -e "${GREEN}✅ jq installé${NC}"
fi

# Vérifier Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker n'est pas installé${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker installé${NC}"

# Vérifier que les conteneurs sont actifs
echo ""
echo "🐳 Vérification des conteneurs Docker..."

containers=("eunomia-api-1" "eunomia-postgres" "eunomia-redis" "eunomia-ollama")
all_running=true

for container in "${containers[@]}"; do
    if docker ps --format '{{.Names}}' | grep -q "^${container}$"; then
        echo -e "${GREEN}✅ $container${NC} - En cours d'exécution"
    else
        echo -e "${RED}❌ $container${NC} - Arrêté ou introuvable"
        all_running=false
    fi
done

if [ "$all_running" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Certains conteneurs ne sont pas actifs${NC}"
    echo ""
    read -p "Voulez-vous démarrer les conteneurs maintenant ? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cd "$PROJECT_ROOT"
        echo "🚀 Démarrage des conteneurs..."
        docker compose -f docker-compose.prod.yml up -d
        echo ""
        echo "⏳ Attente de 30 secondes pour que les services soient prêts..."
        sleep 30
        echo -e "${GREEN}✅ Conteneurs démarrés${NC}"
    else
        echo -e "${RED}❌ Impossible de continuer sans les conteneurs actifs${NC}"
        exit 1
    fi
fi

# ==============================================================================
# ÉTAPE 2 : VÉRIFICATION DE LA SANTÉ DE L'API
# ==============================================================================

print_step "🏥 ÉTAPE 2/6 : Vérification de la santé de l'API"

echo ""
echo "🔍 Test de connexion à l'API..."

API_URL="https://api.lyesbadii.xyz"

health_response=$(curl -s -w "\n%{http_code}" "$API_URL/health" 2>/dev/null || echo -e "\n000")
http_code=$(echo "$health_response" | tail -n1)
body=$(echo "$health_response" | sed '$d')

if [ "$http_code" = "200" ]; then
    echo -e "${GREEN}✅ API accessible (HTTP $http_code)${NC}"
    echo "$body" | jq '.' 2>/dev/null || echo "$body"
else
    echo -e "${RED}❌ API inaccessible (HTTP $http_code)${NC}"
    echo ""
    echo "🔧 Vérifications à faire :"
    echo "   1. Les conteneurs sont-ils actifs ? docker ps"
    echo "   2. NGINX est-il configuré ? docker logs eunomia-nginx"
    echo "   3. Le certificat SSL est-il valide ? curl -k https://localhost/health"
    echo ""
    exit 1
fi

# ==============================================================================
# ÉTAPE 3 : VÉRIFICATION DE LA BASE DE DONNÉES
# ==============================================================================

print_step "🗄️  ÉTAPE 3/6 : Vérification de la base de données"

echo ""
echo "🔍 Connexion à PostgreSQL..."

if docker exec eunomia-postgres psql -U eunomia_user -d eunomia_db -c "SELECT 1;" &>/dev/null; then
    echo -e "${GREEN}✅ Base de données accessible${NC}"
    user_count=$(docker exec eunomia-postgres psql -U eunomia_user -d eunomia_db -t -A -c "SELECT COUNT(*) FROM users;" 2>/dev/null)
    echo "📊 Utilisateurs dans la base : ${user_count:-0}"
else
    echo -e "${RED}❌ Impossible de se connecter à la base de données${NC}"
    exit 1
fi

# ==============================================================================
# ÉTAPE 4 : CRÉATION DU RÉPERTOIRE DE TEST
# ==============================================================================

print_step "📁 ÉTAPE 4/6 : Préparation du répertoire de test"

TEST_DIR="$HOME/eunomia-test-results"
mkdir -p "$TEST_DIR"

echo ""
echo "📂 Répertoire de test créé : $TEST_DIR"
echo "   Les résultats des tests y seront sauvegardés."

# ==============================================================================
# ÉTAPE 5 : VÉRIFICATION DU SCRIPT DE TEST
# ==============================================================================

print_step "🧪 ÉTAPE 5/6 : Vérification du script de test"

TEST_SCRIPT="$SCRIPT_DIR/test-eunomia-api.sh"

if [ -f "$TEST_SCRIPT" ]; then
    echo -e "${GREEN}✅ Script de test trouvé : $TEST_SCRIPT${NC}"
    
    # Rendre exécutable
    chmod +x "$TEST_SCRIPT"
    echo "✅ Permissions d'exécution accordées"
else
    echo -e "${RED}❌ Script de test introuvable : $TEST_SCRIPT${NC}"
    echo ""
    echo "📥 Le script devrait se trouver dans : $SCRIPT_DIR/"
    echo ""
    echo "💡 Solutions :"
    echo "   1. Vérifiez que le fichier test-eunomia-api.sh existe"
    echo "   2. Téléchargez-le depuis Claude si nécessaire"
    echo ""
    exit 1
fi

# ==============================================================================
# ÉTAPE 6 : LANCEMENT DES TESTS
# ==============================================================================

print_step "🚀 ÉTAPE 6/6 : Lancement de la suite de tests complète"

echo ""
echo "📝 Les tests vont :"
echo "   1. Créer automatiquement un utilisateur de test"
echo "   2. Tester tous les endpoints d'authentification"
echo "   3. Tester les endpoints de gestion utilisateur"
echo "   4. Tester l'upload et la gestion de documents"
echo "   5. Générer un rapport de résultats"
echo ""
echo -e "${CYAN}⏱️  Durée estimée : 2-3 minutes${NC}"
echo ""

read -p "▶️  Appuyez sur Entrée pour démarrer les tests..." -r
echo ""

# Lancer les tests et capturer la sortie
LOG_FILE="$TEST_DIR/test-run-$(date +%Y%m%d-%H%M%S).log"

echo "📊 Les résultats seront sauvegardés dans : $LOG_FILE"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Lancer le script de test
"$TEST_SCRIPT" 2>&1 | tee "$LOG_FILE"

TEST_EXIT_CODE=${PIPESTATUS[0]}

# ==============================================================================
# RÉSULTATS FINAUX
# ==============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅✅✅ TESTS TERMINÉS AVEC SUCCÈS ✅✅✅${NC}"
else
    echo -e "${YELLOW}⚠️  TESTS TERMINÉS AVEC DES AVERTISSEMENTS${NC}"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}📊 RÉSUMÉ${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📁 Rapport sauvegardé dans :"
echo "   $LOG_FILE"
echo ""
echo "🔍 Pour consulter les logs :"
echo "   cat $LOG_FILE"
echo ""
echo "📊 Pour voir les utilisateurs créés :"
echo "   docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -c 'SELECT id, email, is_verified, created_at FROM users ORDER BY id DESC LIMIT 5;'"
echo ""
echo "📈 Pour voir les documents uploadés :"
echo "   docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -c 'SELECT id, filename, status, uploaded_at FROM documents ORDER BY id DESC LIMIT 5;'"
echo ""
echo "🗑️  Pour nettoyer les données de test :"
echo "   docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -c \"DELETE FROM users WHERE email LIKE 'test-%@eunomia.legal';\""
echo ""

# Statistiques finales
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${CYAN}📈 STATISTIQUES FINALES${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Compter les tests réussis
success_count=$(grep -c "✅ SUCCESS" "$LOG_FILE" 2>/dev/null || echo "0")
failed_count=$(grep -c "❌ FAILED" "$LOG_FILE" 2>/dev/null || echo "0")

echo "✅ Tests réussis : $success_count"
echo "❌ Tests échoués : $failed_count"
echo ""

if [ "$failed_count" -gt 0 ]; then
    echo -e "${YELLOW}⚠️  Des tests ont échoué. Consultez le rapport pour plus de détails.${NC}"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${GREEN}🎉 Test complet terminé !${NC}"
echo ""
echo "💡 Prochaines étapes :"
echo "   1. Consultez le rapport : cat $LOG_FILE"
echo "   2. Testez l'interface web : https://lyesbadii.xyz"
echo "   3. Créez votre compte personnel via l'API ou l'interface"
echo ""
echo "🆘 Support :"
echo "   - Documentation : https://github.com/votre-repo/docs"
echo "   - Logs backend : docker logs eunomia-backend-api-1"
echo "   - Logs Ollama : docker logs eunomia-ollama"
echo ""

exit $TEST_EXIT_CODE
