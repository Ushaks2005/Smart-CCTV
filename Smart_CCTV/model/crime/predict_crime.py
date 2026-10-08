import random
import time
import logging

def predict_crime_event():
    """
    Simulates a predictive model for crime risk.
    In a real system, this could use an ML model trained on CCTV patterns.
    """

    logging.info("🔮 Crime prediction engine started.")

    while True:
        # Simulate periodic risk evaluation
        time.sleep(10)

        # Randomly simulate prediction
        risk_score = random.uniform(0, 1)

        if risk_score > 0.8:
            logging.warning(f"🚨 High crime probability detected! Risk Score: {risk_score:.2f}")
        else:
            logging.info(f"✅ Normal activity. Risk Score: {risk_score:.2f}")
