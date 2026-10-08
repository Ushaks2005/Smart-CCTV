from flask import Flask, request, jsonify

app = Flask(__name__)

# Example route — you can add more later
@app.route("/alerts", methods=["GET"])
def get_alerts():
    return jsonify({"message": "Alerts API working!"})

@app.route("/alerts", methods=["POST"])
def post_alert():
    data = request.get_json()
    print(f"Received alert: {data}")
    return jsonify({"status": "Alert received!"}), 200

def start_flask_api():
    print("Starting Flask API on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)

# Only runs when this file is executed directly
if __name__ == "__main__":
    start_flask_api()
