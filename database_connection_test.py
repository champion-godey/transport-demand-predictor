import mysql.connector
from mysql.connector import Error


try:
    connection = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="kumasi_transport_ai"
    )

    if connection.is_connected():
        print("========================================")
        print("DATABASE CONNECTION SUCCESSFUL")
        print("========================================")

        print("Host: localhost")
        print("Database: kumasi_transport_ai")
        print("User: root")

except Error as e:
    print("========================================")
    print("DATABASE CONNECTION FAILED")
    print("========================================")

    print("Error:", e)

finally:
    if 'connection' in locals() and connection.is_connected():
        connection.close()
        print("\nDatabase connection closed.")