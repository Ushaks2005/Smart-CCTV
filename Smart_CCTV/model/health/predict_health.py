import random
import time
import logging

def predict_health_event():
    """
    Simulates prediction for health emergencies such as falls or fainting.
    """

    logging.info("🩺 Health prediction engine started.")

    while True:
        time.sleep(10)

        risk_score = random.uniform(0, 1)

        if risk_score > 0.85:
            logging.warning(f"⚕️ Possible health emergency detected! Risk Score: {risk_score:.2f}")
        else:
            logging.info(f"✅ Stable health condition. Risk Score: {risk_score:.2f}")
