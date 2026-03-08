from app import app, index

with app.test_client() as client:
    # Create a test session
    with client.session_transaction() as sess:
        sess['user_id'] = 1
        sess['username'] = 'testuser'
    
    # Make request
    response = client.get('/')
    html = response.get_data(as_text=True)
    print(html[:1500])

