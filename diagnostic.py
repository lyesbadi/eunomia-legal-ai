#!/usr/bin/env python3
"""
Script de diagnostic avancé pour EUNOMIA API
À exécuter: docker exec -it eunomia-api-1 python /app/diagnostic.py
"""
import sys
import os

# Ensure app is in path
sys.path.insert(0, '/app')

def test_basic_imports():
    """Test les imports de base"""
    print("\n" + "="*80)
    print("🔍 TEST 1: IMPORTS DE BASE")
    print("="*80)
    
    try:
        print("✓ Importing FastAPI...")
        from fastapi import FastAPI
        print("  ✅ FastAPI OK")
        
        print("✓ Importing Pydantic...")
        from pydantic import BaseModel
        print("  ✅ Pydantic OK")
        
        print("✓ Importing SQLAlchemy...")
        from sqlalchemy.ext.asyncio import AsyncSession
        print("  ✅ SQLAlchemy OK")
        
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False


def test_app_core():
    """Test les modules core de l'application"""
    print("\n" + "="*80)
    print("🔍 TEST 2: MODULES CORE")
    print("="*80)
    
    success = True
    
    # Test config
    try:
        print("✓ Importing app.core.config...")
        from app.core.config import settings
        print(f"  ✅ Config OK - Environment: {settings.ENVIRONMENT}")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        success = False
    
    # Test database
    try:
        print("✓ Importing app.core.database...")
        from app.core.database import get_db, Base
        print("  ✅ Database OK")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        success = False
    
    # Test security
    try:
        print("✓ Importing app.core.security...")
        from app.core.security import hash_password, verify_password
        print("  ✅ Security OK")
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        success = False
    
    return success


def test_models():
    """Test les models SQLAlchemy"""
    print("\n" + "="*80)
    print("🔍 TEST 3: MODELS SQLALCHEMY")
    print("="*80)
    
    success = True
    
    models = ['user', 'document', 'analysis', 'audit_log']
    for model in models:
        try:
            print(f"✓ Importing app.models.{model}...")
            __import__(f'app.models.{model}')
            print(f"  ✅ {model} OK")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            success = False
    
    return success


def test_schemas():
    """Test les schemas Pydantic"""
    print("\n" + "="*80)
    print("🔍 TEST 4: SCHEMAS PYDANTIC")
    print("="*80)
    
    success = True
    
    schemas = ['auth', 'user', 'document', 'analysis']
    for schema in schemas:
        try:
            print(f"✓ Importing app.schemas.{schema}...")
            __import__(f'app.schemas.{schema}')
            print(f"  ✅ {schema} OK")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success


def test_api_deps():
    """Test les dépendances API"""
    print("\n" + "="*80)
    print("🔍 TEST 5: API DEPENDENCIES")
    print("="*80)
    
    try:
        print("✓ Importing app.api.deps...")
        from app.api.deps import (
            get_current_user,
            get_current_active_user,
            get_audit_logger,
            rate_limit_strict
        )
        print("  ✅ All dependencies OK")
        print(f"  - get_current_user: {get_current_user}")
        print(f"  - rate_limit_strict: {rate_limit_strict}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_api_routes():
    """Test les routes API"""
    print("\n" + "="*80)
    print("🔍 TEST 6: API ROUTES")
    print("="*80)
    
    success = True
    
    routes = {
        'auth': 'app.api.v1.auth',
        'users': 'app.api.v1.users',
        'documents': 'app.api.v1.documents',
        'analyses': 'app.api.v1.analyses',
    }
    
    for name, module_path in routes.items():
        try:
            print(f"✓ Importing {module_path}...")
            module = __import__(module_path, fromlist=['router'])
            router = getattr(module, 'router')
            print(f"  ✅ {name} router OK")
            print(f"    Router routes: {len(router.routes) if hasattr(router, 'routes') else 'N/A'}")
        except Exception as e:
            print(f"  ❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            success = False
    
    return success


def test_api_aggregation():
    """Test l'agrégation des routes"""
    print("\n" + "="*80)
    print("🔍 TEST 7: API AGGREGATION")
    print("="*80)
    
    try:
        print("✓ Importing app.api.v1...")
        from app.api.v1 import api_router
        print(f"  ✅ v1 api_router OK")
        print(f"    Total routes in v1: {len(api_router.routes)}")
        
        print("\n✓ Importing app.api...")
        from app.api import api_router as main_api_router
        print(f"  ✅ main api_router OK")
        print(f"    Total routes in main: {len(main_api_router.routes)}")
        
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_main_app():
    """Test l'application principale"""
    print("\n" + "="*80)
    print("🔍 TEST 8: MAIN APPLICATION")
    print("="*80)
    
    try:
        print("✓ Importing app.main...")
        from app.main import app
        print(f"  ✅ Main app OK")
        print(f"    Total routes registered: {len(app.routes)}")
        
        # List all routes
        print("\n  📋 Registered routes:")
        for route in app.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods'):
                methods = ', '.join(route.methods) if route.methods else 'N/A'
                print(f"    {methods:10} {route.path}")
        
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_route_exists():
    """Vérifie si la route /api/v1/auth/register existe"""
    print("\n" + "="*80)
    print("🔍 TEST 9: ROUTE VERIFICATION")
    print("="*80)
    
    try:
        from app.main import app
        
        target_route = "/api/v1/auth/register"
        found = False
        
        for route in app.routes:
            if hasattr(route, 'path'):
                if route.path == target_route:
                    found = True
                    print(f"  ✅ Route '{target_route}' FOUND!")
                    if hasattr(route, 'methods'):
                        print(f"    Methods: {route.methods}")
                    break
        
        if not found:
            print(f"  ❌ Route '{target_route}' NOT FOUND!")
            print(f"\n  Available auth routes:")
            for route in app.routes:
                if hasattr(route, 'path') and '/auth' in route.path:
                    methods = ', '.join(route.methods) if hasattr(route, 'methods') else 'N/A'
                    print(f"    {methods:10} {route.path}")
            
            return False
        
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Exécute tous les tests"""
    print("="*80)
    print("🚀 EUNOMIA API - DIAGNOSTIC COMPLET")
    print("="*80)
    
    results = {}
    
    # Exécuter tous les tests
    results['basic_imports'] = test_basic_imports()
    results['core'] = test_app_core()
    results['models'] = test_models()
    results['schemas'] = test_schemas()
    results['api_deps'] = test_api_deps()
    results['api_routes'] = test_api_routes()
    results['api_aggregation'] = test_api_aggregation()
    results['main_app'] = test_main_app()
    results['route_check'] = check_route_exists()
    
    # Résumé
    print("\n" + "="*80)
    print("📊 RÉSUMÉ DES TESTS")
    print("="*80)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if failed > 0:
        print(f"\n⚠️  {failed} test(s) failed - Check errors above")
        print("\n💡 PROCHAINES ÉTAPES:")
        print("  1. Vérifier les erreurs d'import ci-dessus")
        print("  2. S'assurer que tous les fichiers sont présents")
        print("  3. Vérifier les dépendances dans requirements.txt")
        print("  4. Rebuild le conteneur si nécessaire")
        sys.exit(1)
    else:
        print("\n✅ Tous les tests sont passés!")
        print("\n🤔 Si la route retourne toujours 404, vérifier:")
        print("  1. La configuration NGINX (reverse proxy)")
        print("  2. Les variables d'environnement")
        print("  3. Redémarrer le conteneur")
        sys.exit(0)


if __name__ == "__main__":
    main()
