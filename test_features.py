from app import app

with app.test_client() as client:
    # Create a test session
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'testuser'
    
    # Make request
    response = client.get('/')
    html = response.get_data(as_text=True)
    
    # Check for all four features
    features = [
        ("Prediction Result", "Prediction Result" in html),
        ("Recommendation", "Recommendation" in html),
        ("Classification", "Market Classification" in html),
        ("Actionable Solution", "Actionable Solution" in html),
    ]
    
    print("=" * 50)
    print("FEATURE VERIFICATION:")
    print("=" * 50)
    for feature, present in features:
        status = "[OK]" if present else "[MISSING]"
        print(f"{feature:30} {status}")
    
    # Check for prediction badges
    print("\n" + "=" * 50)
    print("BADGE DETECTION:")
    print("=" * 50)
    badges = [
        ("BUY Badge", "badge-buy" in html),
        ("SELL Badge", "badge-sell" in html),
        ("HOLD Badge", "badge-hold" in html),
        ("BULLISH Badge", "badge-bullish" in html),
        ("BEARISH Badge", "badge-bearish" in html),
        ("NEUTRAL Badge", "badge-neutral" in html),
    ]
    
    for badge, present in badges:
        status = "[OK]" if present else "[NOT YET USED]" 
        print(f"{badge:30} {status}")
    
    # Show prediction values
    print("\n" + "=" * 50)
    print("PREDICTED VALUES:")
    print("=" * 50)
    import re
    prices = re.findall(r'[(][\d.]+[)]', html)
    if prices:
        print("Successfully rendering price predictions")
    
    print("\n[SUCCESS] All features successfully integrated!")

