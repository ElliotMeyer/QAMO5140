#!/usr/bin/env python3
"""
Test script to verify all data science packages are properly installed and can be imported.
"""

def test_imports():
    """Test importing all main data science packages."""
    
    print("🧪 Testing package imports...\n")
    
    # Test core packages
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print(f"✅ Pandas {pd.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✅ Matplotlib {matplotlib.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print(f"✅ Seaborn {sns.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    # Test machine learning packages
    try:
        import sklearn
        print(f"✅ Scikit-learn {sklearn.__version__} imported successfully")
    except ImportError as e:
        print(f"❌ Scikit-learn import failed: {e}")
        return False
    
    # Test additional packages
    try:
        import plotly.express as px
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"⚠️  Plotly import failed: {e}")
    
    try:
        import jupyter
        print("✅ Jupyter imported successfully")
    except ImportError as e:
        print(f"⚠️  Jupyter import failed: {e}")
    
    try:
        import requests
        print("✅ Requests imported successfully")
    except ImportError as e:
        print(f"⚠️  Requests import failed: {e}")
    
    print("\n🧪 Testing basic functionality...")
    
    # Test NumPy functionality
    arr = np.array([1, 2, 3, 4, 5])
    print(f"✅ NumPy array: {arr}, Mean: {arr.mean():.2f}")
    
    # Test Pandas functionality
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    print(f"✅ Pandas DataFrame created successfully")
    
    # Test Scikit-learn functionality
    from sklearn.ensemble import RandomForestClassifier
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    y = np.array([1, 1, 2, 2])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    clf.fit(X, y)
    print(f"✅ Scikit-learn model trained successfully! Score: {clf.score(X, y):.2f}")
    
    print("\n🎉 All tests passed! Your environment is ready for data science and machine learning!")
    return True

if __name__ == "__main__":
    test_imports()
