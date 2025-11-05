#!/bin/bash

echo "=========================================="
echo " TEST DIRECT DES CONTENEURS API"
echo "=========================================="
echo ""

echo "1️  Test du health check (devrait fonctionner):"
echo "-------------------------------------------"
echo "$ curl http://localhost:8000/health"
echo ""
docker exec -it eunomia-api-1 curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null || echo "Erreur lors du parsing JSON"
echo ""

echo "2️  Test de la route /api/v1/auth/register DIRECTEMENT dans le conteneur:"
echo "------------------------------------------------------------------------"
echo "$ curl -X POST http://localhost:8000/api/v1/auth/register"
echo ""
docker exec -it eunomia-api-1 curl -s -X POST \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"Test123!","password_confirm":"Test123!","full_name":"Test User","gdpr_consent":true}' \
  http://localhost:8000/api/v1/auth/register | python3 -m json.tool 2>/dev/null || echo "Erreur lors du parsing JSON"
echo ""

echo "3️  Liste des routes disponibles (via /debug/routes si activé):"
echo "---------------------------------------------------------------"
echo "$ curl http://localhost:8000/debug/routes"
echo ""
docker exec -it eunomia-api-1 curl -s http://localhost:8000/debug/routes 2>/dev/null | python3 -m json.tool 2>/dev/null || echo "Endpoint /debug/routes non disponible (normal en production)"
echo ""

echo "4️  Vérification de la documentation OpenAPI:"
echo "--------------------------------------------"
echo "$ curl http://localhost:8000/openapi.json | grep '/api/v1/auth'"
echo ""
docker exec -it eunomia-api-1 curl -s http://localhost:8000/openapi.json 2>/dev/null | grep -o '"/api/v1/auth[^"]*"' | head -10 || echo "Aucune route auth trouvée dans OpenAPI"
echo ""

echo "=========================================="
echo "ANALYSE"
echo "=========================================="
echo ""
echo "Si la route retourne 404 MÊME en test direct (sans NGINX):"
echo "    Le problème est dans FastAPI (import/configuration)"
echo "    Exécuter: docker exec -it eunomia-api-1 python /app/diagnostic.py"
echo ""
echo "Si la route fonctionne en test direct mais pas via NGINX:"
echo "    Le problème est dans la configuration NGINX"
echo "    Vérifier: /etc/nginx/conf.d/eunomia.conf"
echo ""
