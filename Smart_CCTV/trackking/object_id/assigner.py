import time
import logging
import random

def assign_ids():
    """
    Simulates assigning unique IDs to detected objects.
    In the future, this can integrate with DeepSORT or YOLO-based ID persistence.
    """
    logging.info("🆔 Object ID assigner started.")

    while True:
        time.sleep(7)
        new_id = random.randint(1000, 9999)
        logging.info(f"🆔 Assigned new tracking ID: {new_id}")
