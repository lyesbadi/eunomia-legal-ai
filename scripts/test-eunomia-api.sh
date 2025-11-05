#!/bin/bash

# ==============================================================================
# EUNOMIA API - Test Suite Complet
# ==============================================================================

set -e

API_URL="https://api.lyesbadii.xyz"
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

print_header() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo -e "${YELLOW}$1${NC}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

test_endpoint() {
    local test_name="$1"
    local method="$2"
    local endpoint="$3"
    local data="$4"
    local auth_header="$5"
    
    echo ""
    echo -e "${YELLOW}🧪 TEST:${NC} $test_name"
    
    if [ -z "$data" ]; then
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$API_URL$endpoint" $auth_header)
    else
        response=$(curl -s -w "\n%{http_code}" -X "$method" "$API_URL$endpoint" \
            -H "Content-Type: application/json" \
            $auth_header \
            -d "$data")
    fi
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | sed '$d')
    
    if [[ "$http_code" == 20* ]]; then
        echo -e "${GREEN}✅ SUCCESS${NC} - HTTP $http_code"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo -e "${RED}❌ FAILED${NC} - HTTP $http_code"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    fi
    
    sleep 1  # Rate limiting
}

# ==============================================================================
# SECTION 1: HEALTH CHECKS
# ==============================================================================

print_header "SECTION 1: HEALTH CHECKS"

test_endpoint \
    "Basic Health Check" \
    "GET" \
    "/health" \
    "" \
    ""

test_endpoint \
    "Detailed Health Check" \
    "GET" \
    "/health/detailed" \
    "" \
    ""

# ==============================================================================
# SECTION 2: AUTHENTICATION
# ==============================================================================

print_header "SECTION 2: AUTHENTICATION"

# Generate unique email for testing
TEST_EMAIL="test-$(date +%s)@eunomia.legal"
TEST_PASSWORD="SecureTest123!"

# 2.1. Register
echo ""
echo "📝 Creating test user: $TEST_EMAIL"
register_response=$(curl -s -X POST "$API_URL/api/v1/auth/register" \
    -H "Content-Type: application/json" \
    -d "{
        \"email\": \"$TEST_EMAIL\",
        \"password\": \"$TEST_PASSWORD\",
        \"password_confirm\": \"$TEST_PASSWORD\",
        \"full_name\": \"Test User Auto\",
        \"gdpr_consent\": true,
        \"language\": \"fr\"
    }")

echo "$register_response" | jq '.'
USER_ID=$(echo "$register_response" | jq -r '.user.id')

# 2.2. Get verification token from DB
echo ""
echo "🔑 Retrieving verification token from database..."
VERIFY_TOKEN=$(docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -t -c \
    "SELECT verification_token FROM users WHERE email='$TEST_EMAIL';" | tr -d ' \r\n')

if [ -n "$VERIFY_TOKEN" ] && [ "$VERIFY_TOKEN" != "" ]; then
    echo "Token found: ${VERIFY_TOKEN:0:20}..."
    
    # 2.3. Verify email
    test_endpoint \
        "Verify Email" \
        "POST" \
        "/api/v1/auth/verify-email" \
        "{\"token\": \"$VERIFY_TOKEN\"}" \
        ""
else
    echo -e "${RED}❌ No verification token found${NC}"
fi

# 2.4. Login
echo ""
echo "🔐 Logging in..."
login_response=$(curl -s -X POST "$API_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{
        \"email\": \"$TEST_EMAIL\",
        \"password\": \"$TEST_PASSWORD\"
    }")

echo "$login_response" | jq '.'

ACCESS_TOKEN=$(echo "$login_response" | jq -r '.access_token')
REFRESH_TOKEN=$(echo "$login_response" | jq -r '.refresh_token')

if [ "$ACCESS_TOKEN" != "null" ] && [ -n "$ACCESS_TOKEN" ]; then
    echo -e "${GREEN}✅ Login successful${NC}"
    AUTH_HEADER="-H \"Authorization: Bearer $ACCESS_TOKEN\""
else
    echo -e "${RED}❌ Login failed${NC}"
    exit 1
fi

# 2.5. Get current user
test_endpoint \
    "Get Current User (/auth/me)" \
    "GET" \
    "/api/v1/auth/me" \
    "" \
    "$AUTH_HEADER"

# 2.6. Refresh token
test_endpoint \
    "Refresh Access Token" \
    "POST" \
    "/api/v1/auth/refresh" \
    "{\"refresh_token\": \"$REFRESH_TOKEN\"}" \
    ""

# 2.7. Logout
test_endpoint \
    "Logout" \
    "POST" \
    "/api/v1/auth/logout" \
    "" \
    "$AUTH_HEADER"

# ==============================================================================
# SECTION 3: USER MANAGEMENT
# ==============================================================================

print_header "SECTION 3: USER MANAGEMENT"

# Re-login for fresh token
login_response=$(curl -s -X POST "$API_URL/api/v1/auth/login" \
    -H "Content-Type: application/json" \
    -d "{\"email\": \"$TEST_EMAIL\", \"password\": \"$TEST_PASSWORD\"}")
ACCESS_TOKEN=$(echo "$login_response" | jq -r '.access_token')
AUTH_HEADER="-H \"Authorization: Bearer $ACCESS_TOKEN\""

# 3.1. Get user profile
test_endpoint \
    "Get User Profile (/users/me)" \
    "GET" \
    "/api/v1/users/me" \
    "" \
    "$AUTH_HEADER"

# 3.2. Get dashboard
test_endpoint \
    "Get User Dashboard with Stats" \
    "GET" \
    "/api/v1/users/me/dashboard" \
    "" \
    "$AUTH_HEADER"

# 3.3. Update profile
test_endpoint \
    "Update User Profile" \
    "PATCH" \
    "/api/v1/users/me" \
    "{\"full_name\": \"Test User UPDATED\", \"language\": \"en\", \"timezone\": \"Europe/Paris\"}" \
    "$AUTH_HEADER"

# 3.4. Export data (GDPR)
echo ""
echo "📥 Exporting user data (GDPR)..."
curl -s -X GET "$API_URL/api/v1/users/me/export" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    --output "/tmp/user_export_$USER_ID.json"

if [ -f "/tmp/user_export_$USER_ID.json" ]; then
    echo -e "${GREEN}✅ Data exported successfully${NC}"
    ls -lh "/tmp/user_export_$USER_ID.json"
else
    echo -e "${RED}❌ Export failed${NC}"
fi

# ==============================================================================
# SECTION 4: DOCUMENT MANAGEMENT
# ==============================================================================

print_header "SECTION 4: DOCUMENT MANAGEMENT"

# 4.1. Create test document
echo ""
echo "📄 Creating test document..."
echo "Ceci est un contrat de test créé automatiquement le $(date)." > /tmp/test_contract.txt

# 4.2. Upload document
echo ""
echo "📤 Uploading document..."
upload_response=$(curl -s -X POST "$API_URL/api/v1/documents/upload" \
    -H "Authorization: Bearer $ACCESS_TOKEN" \
    -F "file=@/tmp/test_contract.txt" \
    -F "title=Contrat de Test Auto" \
    -F "description=Document créé par script de test" \
    -F "language=fr" \
    -F "analyze_immediately=false")

echo "$upload_response" | jq '.'
DOCUMENT_ID=$(echo "$upload_response" | jq -r '.document_id')

if [ "$DOCUMENT_ID" != "null" ] && [ -n "$DOCUMENT_ID" ]; then
    echo -e "${GREEN}✅ Document uploaded (ID: $DOCUMENT_ID)${NC}"
    
    # 4.3. List documents
    test_endpoint \
        "List Documents" \
        "GET" \
        "/api/v1/documents/?skip=0&limit=10" \
        "" \
        "$AUTH_HEADER"
    
    # 4.4. Get document details
    test_endpoint \
        "Get Document Details" \
        "GET" \
        "/api/v1/documents/$DOCUMENT_ID" \
        "" \
        "$AUTH_HEADER"
    
    # 4.5. Update document
    test_endpoint \
        "Update Document Metadata" \
        "PATCH" \
        "/api/v1/documents/$DOCUMENT_ID" \
        "{\"title\": \"Contrat de Test MODIFIÉ\", \"description\": \"Description mise à jour\"}" \
        "$AUTH_HEADER"
    
    # 4.6. Download document
    echo ""
    echo "📥 Downloading document..."
    curl -s -X GET "$API_URL/api/v1/documents/$DOCUMENT_ID/download" \
        -H "Authorization: Bearer $ACCESS_TOKEN" \
        --output "/tmp/downloaded_$DOCUMENT_ID.txt"
    
    if [ -f "/tmp/downloaded_$DOCUMENT_ID.txt" ]; then
        echo -e "${GREEN}✅ Document downloaded successfully${NC}"
        ls -lh "/tmp/downloaded_$DOCUMENT_ID.txt"
    else
        echo -e "${RED}❌ Download failed${NC}"
    fi
    
    # 4.7. Get statistics
    test_endpoint \
        "Get User Statistics" \
        "GET" \
        "/api/v1/documents/statistics" \
        "" \
        "$AUTH_HEADER"
    
    # 4.8. Delete document
    test_endpoint \
        "Delete Document" \
        "DELETE" \
        "/api/v1/documents/$DOCUMENT_ID" \
        "" \
        "$AUTH_HEADER"
else
    echo -e "${RED}❌ Document upload failed${NC}"
fi

# ==============================================================================
# SECTION 5: PASSWORD MANAGEMENT
# ==============================================================================

print_header "SECTION 5: PASSWORD MANAGEMENT"

# 5.1. Request password reset
test_endpoint \
    "Request Password Reset" \
    "POST" \
    "/api/v1/auth/password-reset" \
    "{\"email\": \"$TEST_EMAIL\"}" \
    ""

# 5.2. Get reset token from DB
echo ""
echo "🔑 Retrieving password reset token..."
RESET_TOKEN=$(docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -t -c \
    "SELECT password_reset_token FROM users WHERE email='$TEST_EMAIL';" | tr -d ' \r\n')

if [ -n "$RESET_TOKEN" ] && [ "$RESET_TOKEN" != "" ]; then
    echo "Token found: ${RESET_TOKEN:0:20}..."
    
    # 5.3. Confirm password reset
    test_endpoint \
        "Confirm Password Reset" \
        "POST" \
        "/api/v1/auth/password-reset/confirm" \
        "{\"token\": \"$RESET_TOKEN\", \"new_password\": \"NewPass123!\", \"new_password_confirm\": \"NewPass123!\"}" \
        ""
else
    echo -e "${YELLOW}⚠️  No reset token (user may already be verified)${NC}"
fi

# ==============================================================================
# CLEANUP
# ==============================================================================

print_header "CLEANUP"

echo ""
echo "🧹 Cleaning up test files..."
rm -f /tmp/test_contract.txt
rm -f /tmp/downloaded_*.txt
rm -f /tmp/user_export_*.json

echo -e "${GREEN}✅ Cleanup complete${NC}"

# ==============================================================================
# SUMMARY
# ==============================================================================

print_header "TEST SUMMARY"

echo ""
echo "Test user created: $TEST_EMAIL"
echo "User ID: $USER_ID"
echo ""
echo -e "${GREEN}✅ All tests completed!${NC}"
echo ""
echo "📊 To view logs:"
echo "   docker logs eunomia-backend-api-1 --tail=50"
echo ""
echo "🗑️  To delete test user:"
echo "   docker exec -it eunomia-postgres psql -U eunomia_user -d eunomia_db -c \"DELETE FROM users WHERE email='$TEST_EMAIL';\""
echo ""
