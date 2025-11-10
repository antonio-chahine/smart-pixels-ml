try:
    import qkeras
    print("✅ QKeras is installed.")
    print("Version:", qkeras.__version__)
except ImportError:
    print("❌ QKeras is NOT installed.")
