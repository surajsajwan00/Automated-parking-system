from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://minorproject123:minorproject123@cluster0.leufw9o.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)

# Access a specific database
db = client["licenseplate"]

# Access a specific collection within the database
collection = db["details"]

# Insert a document into the collection
data = {"name": "Gupta", "licenseplate": "AO9BH1674", "emailid": "itsakankshag@gmail.com", "department": "Computer Science"}
collection.insert_one(data)