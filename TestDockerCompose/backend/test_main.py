from fastapi.testclient import TestClient

from main import app

# create testing client
client = TestClient(app)


# test index page
def test_index() :
    '''
    test if index page
    '''
    response = client.get('/')
    assert response.status_code == 200
    assert response.json() == {"message" : "welcome to the Air Paradis API"}


# test 200, correct behavior
def test_correct_input() :
    '''
    test response with correct input : keys, types
    '''
    response = client.post(
        '/predict',
        json={"text" : "this is a test tweet"}
        )
    assert response.status_code == 200
    assert list(response.json().keys()) == ["score", "sentiment"]
    assert type(float(response.json()["score"])) == float
    assert response.json()["sentiment"] in ["negative", "positive"]


# test 400 - empty tweet
def test_empty_input() :
    '''
    test output with an empty tweet
    '''
    response = client.post(
        '/predict',
        json={"text" : ""}
        )
    assert response.status_code == 400
    assert response.json() == {"detail" : "empty input"}


# test 400 - wrong key (not "text")
def test_wrong_input_key() :
    '''
    test output with a wrong key (not "text")
    '''
    response = client.post(
        '/predict',
        json={"wrong_key" : "tweet"}
        )
    assert response.status_code == 400
    assert response.json() == {"detail" : "key should be 'text'"}


# test 400 - wrong value (not string type)
def test_wrong_input_value_type() :
    '''
    test output with wrong tweet type
    '''
    response = client.post(
        '/predict',
        json={"text" : 1}
        )
    assert response.status_code == 400
    assert response.json() == {"detail" : "input value should be in string format"}